# AI in Transportation & Logistics

## Table of Contents
1. [Introduction](#introduction)
2. [Autonomous Vehicle Perception Stack](#autonomous-vehicle-perception-stack)
   - [Camera-Based Object Detection (YOLO, DETR)](#camera-based-object-detection-yolo-detr)
   - [LiDAR Point Cloud Processing (PointNet, VoxelNet)](#lidar-point-cloud-processing-pointnet-voxelnet)
   - [Radar Signal Processing](#radar-signal-processing)
   - [Sensor Fusion Architectures](#sensor-fusion-architectures)
   - [BEV Transformers](#bev-transformers)
3. [Path Planning & Motion Control](#path-planning--motion-control)
   - [A* and Hybrid A*](#a-and-hybrid-a)
   - [RRT and RRT*](#rrt-and-rrt)
   - [Imitation Learning for Driving](#imitation-learning-for-driving)
   - [Model Predictive Control (MPC)](#model-predictive-control-mpc)
4. [Traffic Prediction & Management](#traffic-prediction--management)
   - [Graph Neural Networks for Road Networks](#graph-neural-networks-for-road-networks)
   - [Traffic Flow Forecasting](#traffic-flow-forecasting)
   - [Adaptive Traffic Signal Control](#adaptive-traffic-signal-control)
5. [Predictive Maintenance for Fleets](#predictive-maintenance-for-fleets)
   - [Remaining Useful Life Estimation](#remaining-useful-life-estimation)
   - [Fleet-Level Health Monitoring](#fleet-level-health-monitoring)
6. [Route Optimization & Logistics](#route-optimization--logistics)
   - [Vehicle Routing Problem with OR-Tools](#vehicle-routing-problem-with-or-tools)
   - [Reinforcement Learning for Delivery Routing](#reinforcement-learning-for-delivery-routing)
   - [Last-Mile Delivery Optimization](#last-mile-delivery-optimization)
7. [Drone Delivery Systems](#drone-delivery-systems)
   - [Autonomous Drone Navigation](#autonomous-drone-navigation)
   - [Airspace Traffic Management](#airspace-traffic-management)
   - [Battery-Aware Routing](#battery-aware-routing)
8. [Case Studies](#case-studies)
   - [Waymo Autonomous Driving Stack](#waymo-autonomous-driving-stack)
   - [Tesla Full Self-Driving (FSD)](#tesla-full-self-driving-fsd)
   - [Uber Freight & Logistics Optimization](#uber-freight--logistics-optimization)
   - [Amazon Prime Air Drone Delivery](#amazon-prime-air-drone-delivery)
9. [Cross-References](#cross-references)
10. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

Transportation is undergoing its most significant transformation since the invention of the internal combustion engine. Artificial Intelligence is the driving force behind this revolution — enabling vehicles to perceive and navigate the world autonomously, predicting and optimizing traffic flows, maintaining fleets proactively, and orchestrating complex logistics networks with unprecedented efficiency.

The global AI in transportation market was valued at $3.5 billion in 2023 and is projected to exceed $26 billion by 2030, growing at a CAGR of over 32%. This growth spans multiple sub-domains:

- **Autonomous Vehicles**: Level 4 autonomous driving systems in commercial deployment across multiple cities
- **Traffic Management**: AI-powered adaptive traffic signals that reduce congestion by 15-40%
- **Predictive Maintenance**: Reducing fleet maintenance costs by 25-35% and unplanned downtime by 60-70%
- **Logistics Optimization**: AI route optimization reducing delivery costs by 15-30%

Transportation AI presents unique technical challenges:
1. **Safety-Critical Real-Time Inference**: Latency requirements of 10-100ms for perception and control
2. **Multi-Modal Sensor Fusion**: Combining cameras, LiDAR, radar, and ultrasonic sensors with fundamentally different characteristics
3. **Rare Event Handling**: Edge cases (e.g., pedestrians in unusual poses, construction zones) that are underrepresented in training data
4. **Regulatory Compliance**: Navigation of evolving regulations across jurisdictions
5. **Robustness to Weather**: Perception systems must operate in rain, snow, fog, and darkness

This document provides a deep technical examination of the architectures, algorithms, and deployment patterns powering modern transportation AI systems.

---

## Autonomous Vehicle Perception Stack

The perception stack is the sensory nervous system of an autonomous vehicle. It processes raw sensor data to build a comprehensive understanding of the vehicle's environment — detecting objects, predicting their motion, and mapping the drivable space.

### Camera-Based Object Detection (YOLO, DETR)

Cameras provide dense semantic information at high resolution with low cost. Modern autonomous vehicles use 8-12 cameras covering a 360-degree field of view.

**YOLOv8 for Real-Time Detection**:

```python
import torch
import torch.nn as nn
import numpy as np

class YOLOv8Head(nn.Module):
    """
    YOLOv8 detection head with decoupled classification and regression.
    Used in many autonomous driving perception pipelines.
    """
    
    def __init__(self, in_channels: int = 256, num_classes: int = 80, 
                 reg_max: int = 16, num_anchors: int = 1):
        super().__init__()
        
        # Decoupled head
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        
        # Classification head
        self.cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        
        # Regression head (DFL + IoU)
        self.reg_pred = nn.Conv2d(
            in_channels, 4 * reg_max * num_anchors, 1
        )
        
        self.reg_max = reg_max
    
    def forward(self, x):
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        
        return cls_out, reg_out
    
    def decode_boxes(self, reg_out: torch.Tensor) -> torch.Tensor:
        """Decode DFL distribution into bounding boxes."""
        b, _, h, w = reg_out.shape
        reg_out = reg_out.view(b, 4, self.reg_max, h, w)
        
        # Distribution Focal Loss softmax
        proj = torch.arange(self.reg_max, dtype=torch.float, device=reg_out.device)
        proj = proj.view(1, 1, self.reg_max, 1, 1)
        
        decoded = (torch.softmax(reg_out, dim=2) * proj).sum(dim=2)
        return decoded


class AutonomousVehicleDetector:
    """
    Production-ready object detector for autonomous driving.
    Integrates multiple camera views with temporal context.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        # Multi-scale detection backbone
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()  # FPN/PAN
        self.heads = nn.ModuleList([
            YOLOv8Head(256, num_classes=12),   # P3
            YOLOv8Head(512, num_classes=12),   # P4
            YOLOv8Head(1024, num_classes=12),  # P5
        ])
        self.load_weights(model_path)
        self.eval()
    
    def _build_backbone(self):
        """CSPDarknet backbone for feature extraction."""
        # Simplified representation
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # ... CSPDarknet stages
        )
    
    def _build_neck(self):
        """Feature Pyramid Network with Path Aggregation."""
        return nn.Module()  # Simplified
    
    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> dict:
        """
        Run detection on a single camera frame.
        
        Args:
            image: (3, H, W) tensor, normalized [0, 1]
        
        Returns:
            Dictionary with detections (boxes, scores, classes)
        """
        features = self.backbone(image.unsqueeze(0))
        pyramid = self.neck(features)
        
        detections = []
        for head, feat in zip(self.heads, pyramid):
            cls_out, reg_out = head(feat)
            boxes = head.decode_boxes(reg_out)
            detections.append((boxes, cls_out))
        
        # NMS and post-processing
        return self._post_process(detections)
    
    def _post_process(self, detections: list) -> dict:
        """Non-maximum suppression with class-agnostic NMS."""
        # Simplified: would include NMS, score filtering, etc.
        return {
            'boxes': torch.randn(50, 4),  # Example
            'scores': torch.rand(50),
            'classes': torch.randint(0, 12, (50,)),
            'num_detections': 50
        }


class DETR3D(nn.Module):
    """
    DETR3D: 3D object detection from multi-view images using transformers.
    A key advancement over anchor-based detectors for autonomous driving.
    """
    
    def __init__(self, num_classes: int = 10, num_queries: int = 300,
                 num_cameras: int = 6, d_model: int = 256):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_cameras = num_cameras
        
        # Shared backbone for all cameras
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ... ResNet-like stages
            nn.AdaptiveAvgPool2d((8, 22))  # BEV grid
        )
        
        # Camera-specific embeddings
        self.camera_embed = nn.Embedding(num_cameras, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Object queries (learnable anchor points in 3D space)
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 3D reference points (initialized in bird's eye view grid)
        self.ref_points = nn.Parameter(self._init_ref_points())
        
        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # (cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy)
        )
    
    def _init_ref_points(self):
        """Initialize reference points in BEV space (uniform grid)."""
        n = int(np.sqrt(self.num_queries))
        x = np.linspace(-50, 50, n)  # 50m range
        y = np.linspace(-25, 25, n)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel(), np.zeros(n*n)], axis=1)
        return torch.FloatTensor(points[:self.num_queries])
    
    def forward(self, multi_view_images: list) -> dict:
        """
        multi_view_images: list of 6 tensors (B, 3, H, W)
        """
        batch_size = multi_view_images[0].size(0)
        
        # Extract features from each camera
        camera_features = []
        for i, img in enumerate(multi_view_images):
            feat = self.backbone(img)
            feat = feat + self.camera_embed(torch.tensor([i], device=img.device))
            camera_features.append(feat)
        
        # Fuse multi-camera features
        fused = torch.stack(camera_features).mean(dim=0)
        fused = fused.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        
        # Transformer decoder
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        memory = fused
        decoded = self.decoder(query, memory)
        
        # Predictions
        class_logits = self.class_head(decoded)
        bbox_params = self.bbox_head(decoded)
        
        # Add reference points to get absolute positions
        bbox_params[:, :, :3] = bbox_params[:, :, :3] + self.ref_points.unsqueeze(0)
        
        return {
            'class_logits': class_logits,
            'bbox_3d': bbox_params
        }
```

### LiDAR Point Cloud Processing (PointNet, VoxelNet)

LiDAR (Light Detection and Ranging) provides accurate 3D geometry measurements, essential for precise localization and obstacle detection.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointNetPlusPlus(nn.Module):
    """
    PointNet++ for direct point cloud processing.
    Used for 3D object detection and segmentation in autonomous driving.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Set abstraction levels
        self.sa1 = self._set_abstraction(512, 0.2, 3, [3, 64, 128])
        self.sa2 = self._set_abstraction(128, 0.4, 128, [128, 128, 256])
        self.sa3 = self._set_abstraction(None, None, 256, [256, 512, 1024])
        
        # Feature propagation for segmentation
        self.fp3 = self._feature_propagation(1024, 512, [512, 256])
        self.fp2 = self._feature_propagation(256, 256, [256, 128])
        self.fp1 = self._feature_propagation(128, 64, [128, 128, 128])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )
    
    def _set_abstraction(self, npoint, radius, in_ch, mlp):
        """Set abstraction layer: sampling, grouping, PointNet."""
        return nn.Module()  # Simplified
    
    def _feature_propagation(self, in_ch1, in_ch2, mlp):
        """Feature propagation layer for upsampling."""
        return nn.Module()
    
    def forward(self, points):
        """
        points: (B, N, 3) — raw point cloud
        """
        l1_xyz, l1_points = self.sa1(points, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Classification (global feature)
        global_feat = l3_points.view(points.size(0), -1)
        class_pred = self.classifier(global_feat)
        
        # Segmentation (per-point prediction)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(points, l1_xyz, None, l1_points)
        
        seg_pred = self.seg_head(l0_points.transpose(1, 2))
        
        return {
            'classification': class_pred,
            'segmentation': seg_pred
        }


class VoxelNet(nn.Module):
    """
    VoxelNet: Point cloud processing via 3D convolutions after voxelization.
    Used in Waymo's perception pipeline.
    """
    
    def __init__(self, voxel_size: tuple = (0.2, 0.2, 0.15),
                 point_cloud_range: tuple = (-75.2, -75.2, -2, 75.2, 75.2, 4),
                 max_points_per_voxel: int = 10):
        super().__init__()
        
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(point_cloud_range)
        self.max_points = max_points_per_voxel
        
        # Voxel Feature Encoding (VFE) layer
        self.vfe = nn.Sequential(
            nn.Linear(7, 32),  # (x, y, z, intensity, r, x_mean, y_mean, z_mean)
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU()
        )
        
        # 3D convolutional middle layers
        self.conv3d = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        
        # Region Proposal Network (RPN) for detection
        self.rpn = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Detection heads
        self.class_head = nn.Conv2d(256, 2, 1)  # Vehicle, pedestrian, cyclist
        self.reg_head = nn.Conv2d(256, 10, 1)   # 3D box regression
    
    def voxelize(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert raw point cloud to voxel grid.
        points: (B, N, 3)
        Returns: (B, D, H, W, features)
        """
        # Simplified voxelization
        # Clip to range
        points = torch.clamp(points, 
                             points.new_tensor(self.pc_range[:3]),
                             points.new_tensor(self.pc_range[3:]))
        
        # Quantize to voxel indices
        voxel_indices = ((points - points.new_tensor(self.pc_range[:3])) 
                         / points.new_tensor(self.voxel_size)).long()
        
        return voxel_indices
    
    def forward(self, points: torch.Tensor) -> dict:
        """
        points: (B, N, 3) raw LiDAR point cloud
        """
        # Voxelization
        voxels = self.voxelize(points)
        
        # VFE
        vfe_out = self.vfe(voxels)
        vfe_out = vfe_out.max(dim=2)[0]  # Max pooling over points in voxel
        
        # 3D convolutions
        conv_out = self.conv3d(vfe_out)
        
        # Collapse height dimension to BEV
        bev = conv_out.max(dim=2)[0]
        
        # RPN
        rpn_out = self.rpn(bev)
        
        # Detection output
        class_pred = self.class_head(rpn_out)
        reg_pred = self.reg_head(rpn_out)
        
        return {
            'class_logits': class_pred,
            'box_regression': reg_pred
        }


class PointPillar(nn.Module):
    """
    PointPillar: Efficient LiDAR processing using pillars (vertical columns)
    instead of full 3D voxels. Significantly more efficient than VoxelNet.
    Used in many production autonomous driving systems.
    """
    
    def __init__(self, 
                 pillar_size: float = 0.2,
                 max_pillars: int = 12000,
                 max_points_per_pillar: int = 100,
                 x_range: tuple = (-75.2, 75.2),
                 y_range: tuple = (-75.2, 75.2),
                 z_range: tuple = (-3, 3)):
        super().__init__()
        
        self.pillar_size = pillar_size
        self.max_pillars = max_pillars
        self.max_points = max_points_per_pillar
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        # Number of pillars along each axis
        self.nx = int((x_range[1] - x_range[0]) / pillar_size)
        self.ny = int((y_range[1] - y_range[0]) / pillar_size)
        
        # Pillar Feature Network
        self.pfn = nn.Sequential(
            nn.Linear(10, 64),  # (x, y, z, intensity, x_c, y_c, z_c, x_m, y_m, z_m)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 2D backbone (similar to VoxelNet's RPN)
        self.backbone_2d = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Detection heads
        self.class_head = nn.Conv2d(256, 3, 1)  # Vehicle, pedestrian, cyclist
        self.reg_head = nn.Conv2d(256, 7, 1)    # (x, y, z, w, l, h, yaw)
    
    def forward(self, points: torch.Tensor) -> dict:
        """
        points: (B, N, 4) — (x, y, z, intensity)
        """
        batch_size = points.size(0)
        
        # Create pillars (simplified)
        pillar_features = self._create_pillars(points)
        
        # PFN
        pfn_out = self.pfn(pillar_features)
        pfn_out = pfn_out.max(dim=1)[0]  # Max over points in pillar
        
        # Reshape to BEV grid
        bev = pfn_out.view(batch_size, 64, self.ny, self.nx)
        
        # 2D backbone
        features = self.backbone_2d(bev)
        
        # Output
        class_pred = self.class_head(features)
        reg_pred = self.reg_head(features)
        
        return {
            'class_logits': class_pred,
            'box_regression': reg_pred
        }
    
    def _create_pillars(self, points):
        """Create pillar representation from point cloud."""
        # Simplified: would create pillar index map and scatter points
        return points.unsqueeze(1)  # Placeholder
```

### Sensor Fusion Architectures

Fusing data from cameras, LiDAR, radar, and ultrasonics is one of the most critical components of the autonomous driving stack.

```python
class SensorFusionTransformer(nn.Module):
    """
    Transformer-based sensor fusion for autonomous driving.
    Fuses camera, LiDAR, and radar features using cross-attention.
    """
    
    def __init__(self, 
                 cam_channels: int = 256,
                 lidar_channels: int = 128,
                 radar_channels: int = 64,
                 d_model: int = 256):
        super().__init__()
        
        # Project each modality to common dimension
        self.cam_proj = nn.Linear(cam_channels, d_model)
        self.lidar_proj = nn.Linear(lidar_channels, d_model)
        self.radar_proj = nn.Linear(radar_channels, d_model)
        
        # Cross-attention fusion layers
        self.cross_attn_cl = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.cross_attn_cr = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.cross_attn_lr = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Self-attention for global context
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                 camera_features: torch.Tensor,
                 lidar_features: torch.Tensor,
                 radar_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-modal features into unified representation.
        
        Args:
            camera_features: (B, N_cam, D_cam)
            lidar_features: (B, N_lid, D_lid)
            radar_features: (B, N_rad, D_rad)
        
        Returns:
            fused_features: (B, N_cam + N_lid + N_rad, D_model)
        """
        # Project to common space
        cam = self.cam_proj(camera_features)
        lid = self.lidar_proj(lidar_features)
        rad = self.radar_proj(radar_features)
        
        # Cross-attention: Camera <-> LiDAR
        cam_fused, _ = self.cross_attn_cl(cam, lid, lid)
        lid_fused, _ = self.cross_attn_cl(lid, cam, cam)
        
        # Camera <-> Radar
        cam_fused, _ = self.cross_attn_cr(cam_fused, rad, rad)
        rad_fused, _ = self.cross_attn_cr(rad, cam_fused, cam_fused)
        
        # LiDAR <-> Radar
        lid_fused, _ = self.cross_attn_lr(lid_fused, rad_fused, rad_fused)
        rad_fused, _ = self.cross_attn_lr(rad_fused, lid_fused, lid_fused)
        
        # Concatenate and apply self-attention
        fused = torch.cat([cam_fused, lid_fused, rad_fused], dim=1)
        fused, _ = self.self_attn(fused, fused, fused)
        
        return self.layer_norm(self.output_proj(fused))
```

### BEV Transformers

Bird's Eye View (BEV) representations have become the dominant paradigm in autonomous driving perception, converting perspective camera views into a top-down representation that aligns naturally with LiDAR and planning:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFormer(nn.Module):
    """
    BEVFormer: Unified BEV transformer for autonomous driving perception.
    Generates BEV features from multi-view cameras using deformable attention.
    
    Key innovation: Pre-defined BEV queries interact with multi-camera features
    through spatio-temporal attention.
    """
    
    def __init__(self, 
                 num_cameras: int = 6,
                 bev_h: int = 200,  # BEV grid height
                 bev_w: int = 200,  # BEV grid width
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_levels: int = 4):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.d_model = d_model
        self.num_levels = num_levels
        
        # BEV queries (learnable)
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, d_model))
        
        # Positional embeddings for BEV grid
        self.bev_pos = self._build_bev_positional_encoding()
        
        # Camera-aware deformable attention
        self.attn_layers = nn.ModuleList([
            BEVCrossAttention(d_model, num_heads, num_levels)
            for _ in range(num_layers)
        ])
        
        # Temporal self-attention
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
    
    def _build_bev_positional_encoding(self):
        """2D positional encoding for BEV grid."""
        pos_h = torch.arange(self.bev_h, dtype=torch.float)
        pos_w = torch.arange(self.bev_w, dtype=torch.float)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Sinusoidal encoding
        d_model_half = self.d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * 
                            -(np.log(10000.0) / d_model_half))
        
        pe = torch.zeros(self.bev_h, self.bev_w, self.d_model)
        pe[:, :, 0:d_model_half:2] = torch.sin(grid_h.unsqueeze(-1) * div_term)
        pe[:, :, 1:d_model_half:2] = torch.cos(grid_h.unsqueeze(-1) * div_term)
        pe[:, :, d_model_half::2] = torch.sin(grid_w.unsqueeze(-1) * div_term)
        pe[:, :, d_model_half+1::2] = torch.cos(grid_w.unsqueeze(-1) * div_term)
        
        return pe.view(-1, self.d_model)
    
    def forward(self, 
                 multi_view_features: list,
                 prev_bev: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            multi_view_features: list of (B, N, C, H, W) for each camera
            prev_bev: (B, H*W, D) previous frame's BEV features
        
        Returns:
            BEV features: (B, H*W, D)
        """
        batch_size = multi_view_features[0].size(0)
        
        # Initialize BEV queries
        bev_queries = self.bev_queries.unsqueeze(0).expand(batch_size, -1, -1)
        bev_queries = bev_queries + self.bev_pos.unsqueeze(0)
        
        for i in range(len(self.attn_layers)):
            # Temporal self-attention
            if prev_bev is not None:
                temp_out, _ = self.temporal_attn[i](
                    bev_queries, prev_bev, prev_bev
                )
                bev_queries = self.norm1[i](bev_queries + temp_out)
            
            # Cross-attention with multi-view features
            cross_out = self.attn_layers[i](
                bev_queries, multi_view_features
            )
            bev_queries = self.norm2[i](bev_queries + cross_out)
            
            # FFN
            ffn_out = self.ffn[i](bev_queries)
            bev_queries = self.norm3[i](bev_queries + ffn_out)
        
        return bev_queries


class BEVCrossAttention(nn.Module):
    """
    Deformable cross-attention from BEV queries to multi-view camera features.
    Each BEV query samples features from relevant image regions using
    pre-computed projection matrices.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_levels: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        
        # Offset prediction
        self.offset_conv = nn.Linear(d_model, num_heads * 2 * num_levels * 6)
        
        # Attention weights
        self.attn_weight = nn.Linear(d_model, num_heads * num_levels * 6)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, 
                 bev_queries: torch.Tensor,
                 multi_view_features: list) -> torch.Tensor:
        """
        Args:
            bev_queries: (B, H*W, D)
            multi_view_features: list of (B, C, H_cam, W_cam) per camera
        """
        # Predict sampling offsets
        offsets = self.offset_conv(bev_queries)
        
        # Predict attention weights
        attn_weights = self.attn_weight(bev_queries)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Sample features from camera views
        # (simplified — actual implementation uses grid_sample)
        sampled_features = torch.zeros_like(bev_queries)
        
        # Output
        out = self.output_proj(sampled_features)
        return out
```

---

## Path Planning & Motion Control

The planning stack translates perception into action, computing safe and efficient trajectories from the vehicle's current state to a goal.

### A* and Hybrid A*

```python
import heapq
import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass

@dataclass
class Node:
    """Search node for path planning."""
    x: float
    y: float
    theta: float  # heading angle
    g: float  # cost from start
    h: float  # heuristic cost to goal
    f: float  # total cost (g + h)
    parent: 'Node' = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return (abs(self.x - other.x) < 0.01 and 
                abs(self.y - other.y) < 0.01 and
                abs(self.theta - other.theta) < 0.1)
    
    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2), round(self.theta, 1)))


class HybridAStar:
    """
    Hybrid A* path planner used in autonomous driving.
    Combines continuous state-space search with Reeds-Shepp curves
    for efficient path planning in complex environments.
    """
    
    def __init__(self, 
                 grid_resolution: float = 0.5,
                 steering_resolution: float = 5.0,  # degrees
                 max_steering_angle: float = 30.0,   # degrees
                 wheelbase: float = 2.7,  # meters
                 vehicle_length: float = 4.5,
                 vehicle_width: float = 1.8):
        
        self.resolution = grid_resolution
        self.steering_res = np.radians(steering_resolution)
        self.max_steering = np.radians(max_steering_angle)
        self.wheelbase = wheelbase
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        
        # Motion primitives
        self.motion_primitives = self._generate_motion_primitives()
    
    def _generate_motion_primitives(self) -> List[Tuple[float, float]]:
        """Generate set of discrete steering actions."""
        primitives = []
        for steering in np.arange(-self.max_steering, 
                                   self.max_steering + self.steering_res,
                                   self.steering_res):
            primitives.append((1.0, steering))  # (velocity, steering_angle)
        return primitives
    
    def plan(self, 
             start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             obstacle_map: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Hybrid A* path planning.
        
        Args:
            start: (x, y, theta) in meters and radians
            goal: (x, y, theta)
            obstacle_map: binary grid (0=free, 1=occupied)
        
        Returns:
            Path as list of (x, y, theta) waypoints
        """
        start_node = Node(start[0], start[1], start[2], 0, 0, 0)
        goal_node = Node(goal[0], goal[1], goal[2], 0, 0, 0)
        
        start_node.h = self._heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        
        open_set = [start_node]
        closed_set: Set[Node] = set()
        
        # Grid for dead-end detection
        visited_grid = set()
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Check if we've visited this grid cell before
            grid_key = (int(current.x / self.resolution),
                        int(current.y / self.resolution))
            if grid_key in visited_grid:
                continue
            visited_grid.add(grid_key)
            
            # Check if close to goal
            if (abs(current.x - goal_node.x) < 1.0 and
                abs(current.y - goal_node.y) < 1.0 and
                abs(self._normalize_angle(current.theta - goal_node.theta)) < 0.3):
                
                # Try Reeds-Shepp connection
                rs_path = self._reeds_shepp_path(current, goal_node)
                if rs_path:
                    return self._reconstruct_path(current, rs_path)
            
            # Expand motion primitives
            for velocity, steering in self.motion_primitives:
                new_node = self._simulate_motion(current, velocity, steering)
                
                # Collision check
                if self._check_collision(new_node, obstacle_map):
                    continue
                
                # Calculate costs
                step_cost = self._motion_cost(new_node, current, obstacle_map)
                new_node.g = current.g + step_cost
                new_node.h = self._heuristic(new_node, goal_node)
                new_node.f = new_node.g + new_node.h
                new_node.parent = current
                
                # Check if better path exists
                if new_node in closed_set:
                    continue
                
                heapq.heappush(open_set, new_node)
        
        return []  # No path found
    
    def _simulate_motion(self, node: Node, velocity: float,
                          steering: float) -> Node:
        """Simulate vehicle motion using bicycle model."""
        dt = 0.5  # Time step (seconds)
        
        # Bicycle model
        x = node.x + velocity * np.cos(node.theta) * dt
        y = node.y + velocity * np.sin(node.theta) * dt
        theta = node.theta + (velocity / self.wheelbase) * np.tan(steering) * dt
        theta = self._normalize_angle(theta)
        
        return Node(x, y, theta, 0, 0, 0)
    
    def _normalize_angle(self, theta: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    def _heuristic(self, node: Node, goal: Node) -> float:
        """Non-holonomic-without-obstacles heuristic using Reeds-Shepp."""
        # Use Euclidean distance as admissible heuristic
        return np.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)
    
    def _check_collision(self, node: Node, obstacle_map: np.ndarray) -> bool:
        """Check if the vehicle footprint collides with obstacles."""
        # Vehicle footprint corners
        half_l = self.vehicle_length / 2
        half_w = self.vehicle_width / 2
        cos_t = np.cos(node.theta)
        sin_t = np.sin(node.theta)
        
        corners = [
            (node.x + cos_t * half_l - sin_t * half_w,
             node.y + sin_t * half_l + cos_t * half_w),
            (node.x + cos_t * half_l + sin_t * half_w,
             node.y + sin_t * half_l - cos_t * half_w),
            (node.x - cos_t * half_l + sin_t * half_w,
             node.y - sin_t * half_l - cos_t * half_w),
            (node.x - cos_t * half_l - sin_t * half_w,
             node.y - sin_t * half_l + cos_t * half_w),
        ]
        
        # Convert to grid coordinates
        for cx, cy in corners:
            gx = int(cx / self.resolution)
            gy = int(cy / self.resolution)
            if (0 <= gx < obstacle_map.shape[0] and
                0 <= gy < obstacle_map.shape[1]):
                if obstacle_map[gx, gy] > 0.5:
                    return True
        
        return False
    
    def _motion_cost(self, new_node: Node, current: Node,
                      obstacle_map: np.ndarray) -> float:
        """Compute cost for a motion primitive."""
        # Base cost: distance
        dist = np.sqrt((new_node.x - current.x)**2 + 
                       (new_node.y - current.y)**2)
        
        # Steering cost (penalize large steering angles)
        steering_cost = 0.1 * abs(self._normalize_angle(
            new_node.theta - current.theta))
        
        # Reverse cost (penalize reversing)
        reverse_cost = 0
        
        # Obstacle proximity cost
        proximity_cost = 0
        
        return dist + steering_cost + reverse_cost + proximity_cost
    
    def _reeds_shepp_path(self, start: Node, goal: Node) -> List[Node]:
        """Compute Reeds-Shepp curve connecting two configurations."""
        # Simplified: return straight-line path
        return [start, goal]
    
    def _reconstruct_path(self, node: Node, 
                           rs_path: List[Node]) -> List[Tuple[float, float, float]]:
        """Reconstruct path from start to goal."""
        path = []
        while node:
            path.append((node.x, node.y, node.theta))
            node = node.parent
        path.reverse()
        
        # Append RS curve
        for p in rs_path[1:]:
            path.append((p.x, p.y, p.theta))
        
        # Smooth path
        return self._smooth_path(path)
    
    def _smooth_path(self, path: List[Tuple]) -> List[Tuple]:
        """Apply B-spline or conjugate gradient smoothing."""
        return path  # Simplified
```

### Imitation Learning for Driving

Imitation learning (IL) learns driving policies directly from human demonstration data:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalImitationModel(nn.Module):
    """
    Conditional Imitation Learning (CIL) for end-to-end driving.
    
    Architecture:
    - Perception encoder (CNN backbone)
    - Route planner (learns navigation commands)
    - Conditional controller (speed, steering conditioned on command)
    
    Reference: Codevilla et al., "End-to-end Driving via Conditional Imitation Learning"
    """
    
    def __init__(self, num_commands: int = 4,  # follow, left, right, straight
                 speed_bins: int = 20):
        super().__init__()
        
        # Perception backbone (ResNet-34)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._resnet_block(64, 64, 3),
            self._resnet_block(64, 128, 4, stride=2),
            self._resnet_block(128, 256, 6, stride=2),
            self._resnet_block(256, 512, 3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Speed encoder
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Command embedding
        self.command_embed = nn.Embedding(num_commands, 32)
        
        # Latent fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Control heads (one per command)
        self.steering_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_commands)
        ])
        
        self.speed_heads = nn.ModuleList([
            nn.Linear(256, speed_bins) for _ in range(num_commands)
        ])
        
        # Speed bin boundaries (0-30 m/s)
        self.register_buffer('speed_boundaries', 
                            torch.linspace(0, 30, speed_bins))
    
    def _resnet_block(self, in_ch, out_ch, num_blocks, stride=1):
        """ResNet bottleneck block."""
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, 
                image: torch.Tensor,
                speed: torch.Tensor,
                command: torch.Tensor) -> dict:
        """
        Args:
            image: (B, 3, H, W)
            speed: (B, 1) current speed in m/s
            command: (B,) discrete command (0=follow, 1=left, etc.)
        
        Returns:
            steering: (B,) steering angle in radians
            target_speed: (B,) target speed in m/s
        """
        # Perception features
        vision_feat = self.backbone(image)
        
        # Speed features
        speed_feat = self.speed_encoder(speed)
        
        # Command features
        cmd_feat = self.command_embed(command)
        
        # Fusion
        fused = torch.cat([vision_feat, speed_feat, cmd_feat], dim=1)
        latent = self.fusion(fused)
        
        # Conditional control outputs
        batch_size = image.size(0)
        steering = torch.zeros(batch_size, device=image.device)
        target_speed = torch.zeros(batch_size, device=image.device)
        
        for i in range(batch_size):
            cmd = command[i].item()
            steering[i] = self.steering_heads[cmd](latent[i:i+1])
            speed_logits = self.speed_heads[cmd](latent[i:i+1])
            speed_probs = F.softmax(speed_logits, dim=1)
            target_speed[i] = (speed_probs * self.speed_boundaries).sum()
        
        return {
            'steering': steering,
            'target_speed': target_speed
        }
    
    def compute_loss(self, prediction: dict, target: dict) -> dict:
        """Compute imitation learning loss."""
        steering_loss = F.mse_loss(prediction['steering'], target['steering'])
        speed_loss = F.mse_loss(prediction['target_speed'], target['target_speed'])
        
        total_loss = steering_loss + speed_loss
        
        return {
            'total_loss': total_loss,
            'steering_loss': steering_loss,
            'speed_loss': speed_loss
        }
```

---

## Traffic Prediction & Management

### Graph Neural Networks for Road Networks

Traffic prediction is inherently a graph-structured problem — road networks naturally form graphs where intersections are nodes and road segments are edges:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrafficGCN(nn.Module):
    """
    Graph Convolutional Network for traffic speed/flow prediction.
    
    Models traffic propagation through the road network using
    spectral graph convolutions.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int = 3,  # speed, flow, occupancy
                 hidden_dim: int = 64,
                 output_horizon: int = 12,  # predict next 12 time steps
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.output_horizon = output_horizon
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(
            GraphConvLayer(input_dim, hidden_dim)
        )
        for _ in range(num_layers - 1):
            self.gcn_layers.append(
                GraphConvLayer(hidden_dim, hidden_dim)
            )
        
        # Temporal convolution (along time axis)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # predict traffic speed
        )
    
    def forward(self, 
                x: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) — traffic time series
            adj_matrix: (N, N) — road network adjacency
        
        Returns:
            predictions: (B, output_horizon, N, 1)
        """
        B, T, N, F = x.shape
        
        # Graph convolutions
        h = x
        for gcn in self.gcn_layers:
            h = gcn(h, adj_matrix)
        
        # Temporal modeling
        h = h.permute(0, 2, 3, 1)  # (B, N, F, T)
        h = h.reshape(B * N, -1, T)  # (B*N, F, T)
        h = self.temporal_conv(h)
        h = h.reshape(B, N, -1, T).permute(0, 3, 1, 2)  # (B, T, N, F)
        
        # Output projection for each time step
        predictions = []
        for t in range(T):
            pred = self.output_layer(h[:, t, :, :])  # (B, N, 1)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (B, T, N, 1)
        
        # Select only the output horizon
        return predictions[:, -self.output_horizon:, :, :]


class GraphConvLayer(nn.Module):
    """
    Spectral graph convolution using Chebyshev polynomial approximation.
    """
    
    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, F)
        adj: (N, N) normalized adjacency matrix
        """
        B, T, N, F = x.shape
        
        # Compute Laplacian
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(dim=1) + 1e-10))
        L = torch.eye(N, device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Chebyshev polynomials
        Tx_0 = x  # T_0(L) * x = x
        output = torch.matmul(Tx_0, self.weight[0])
        
        if self.K > 1:
            Tx_1 = torch.matmul(L, x.reshape(-1, N, F)).reshape(B, T, N, F)
            output = output + torch.matmul(Tx_1, self.weight[1])
        
        for k in range(2, self.K):
            Tx_k = 2 * torch.matmul(L, Tx_1.reshape(-1, N, F)).reshape(B, T, N, F) - Tx_0
            output = output + torch.matmul(Tx_k, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_k
        
        return output + self.bias


class DiffusionConvolutionalRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN).
    State-of-the-art for traffic forecasting. Uses diffusion convolution
    within a GRU cell to model spatio-temporal dependencies.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int = 3,
                 hidden_dim: int = 64,
                 output_horizon: int = 12,
                 num_rnn_layers: int = 2,
                 diffusion_steps: int = 2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_horizon = output_horizon
        self.num_rnn_layers = num_rnn_layers
        self.diffusion_steps = diffusion_steps
        
        # Encoder
        self.encoder_cells = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim,
                     hidden_dim, num_nodes, diffusion_steps)
            for i in range(num_rnn_layers)
        ])
        
        # Decoder (same structure, different weights)
        self.decoder_cells = nn.ModuleList([
            DCGRUCell(hidden_dim, hidden_dim, num_nodes, diffusion_steps)
            for _ in range(num_rnn_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, 
                x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, N, F)
        adj: (N, N) adjacency matrix
        """
        B, T_in, N, F = x.shape
        
        # Encoder
        h_t = [torch.zeros(B, N, self.hidden_dim, device=x.device)
               for _ in range(self.num_rnn_layers)]
        
        for t in range(T_in):
            inp = x[:, t, :, :]
            for layer in range(self.num_rnn_layers):
                h_t[layer] = self.encoder_cells[layer](
                    inp, h_t[layer], adj
                )
                inp = h_t[layer]
        
        # Decoder (using scheduled sampling during training)
        outputs = []
        decoder_input = h_t[-1]
        
        for t in range(self.output_horizon):
            for layer in range(self.num_rnn_layers):
                h_t[layer] = self.decoder_cells[layer](
                    decoder_input, h_t[layer], adj
                )
                decoder_input = h_t[layer]
            
            out = self.output_proj(h_t[-1])  # (B, N, 1)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # (B, T_out, N, 1)


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU cell.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_nodes: int, diffusion_steps: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.diffusion_steps = diffusion_steps
        
        # Diffusion convolution weights for GRU gates
        self.W_input = nn.Parameter(torch.FloatTensor(input_dim, 3 * hidden_dim))
        self.W_hidden = nn.Parameter(torch.FloatTensor(hidden_dim, 3 * hidden_dim))
        self.diff_conv = nn.Parameter(torch.FloatTensor(
            diffusion_steps + 1, 2, 3 * hidden_dim
        ))
        
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_input)
        nn.init.xavier_uniform_(self.W_hidden)
        nn.init.xavier_uniform_(self.diff_conv)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, F)
        h: (B, N, D)
        adj: (N, N)
        """
        B, N, _ = x.shape
        
        # Input and hidden projections
        x_proj = torch.matmul(x, self.W_input)
        h_proj = torch.matmul(h, self.W_hidden)
        
        # Diffusion convolution
        supports = [adj, adj.transpose(0, 1)]  # Forward and backward
        
        diff_out = 0
        support_out = [torch.matmul(s, x_proj.permute(1, 0, 2).reshape(N, -1))
                       for s in supports]
        support_out = [s.reshape(N, B, -1).permute(1, 0, 2) for s in support_out]
        
        diff_out = (self.diff_conv[0, 0] * support_out[0] + 
                    self.diff_conv[0, 1] * support_out[1])
        
        for k in range(1, self.diffusion_steps + 1):
            for s_idx, s in enumerate(supports):
                support_out[s_idx] = torch.matmul(
                    s, support_out[s_idx].permute(1, 0, 2).reshape(N, -1)
                ).reshape(N, B, -1).permute(1, 0, 2)
                
                diff_out = diff_out + (self.diff_conv[k, s_idx] * support_out[s_idx])
        
        # GRU gates
        total = x_proj + h_proj + diff_out + self.bias
        
        r, z, n = torch.chunk(total, 3, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)
        
        new_h = (1 - z) * n + z * h
        return new_h
```

---

## Predictive Maintenance for Fleets

### Remaining Useful Life Estimation

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class FleetRULPredictor:
    """
    Remaining Useful Life (RUL) prediction for fleet vehicles.
    Combines multiple sensor modalities with maintenance history.
    """
    
    def __init__(self, num_sensors: int = 50, seq_length: int = 100):
        self.num_sensors = num_sensors
        self.seq_length = seq_length
        
        # Multi-head model for different component types
        self.engine_model = self._build_rul_model()
        self.transmission_model = self._build_rul_model()
        self.brake_model = self._build_rul_model()
        self.battery_model = self._build_rul_model()  # For EVs
    
    def _build_rul_model(self) -> nn.Module:
        """Build a RUL estimation model using attention and TCN."""
        return nn.Sequential(
            nn.Conv1d(self.num_sensors, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Positive RUL
        )
    
    def predict_rul(self, 
                    sensor_data: Dict[str, np.ndarray],
                    vehicle_type: str = 'truck') -> Dict[str, float]:
        """
        Predict RUL for each major component.
        
        sensor_data: Dict mapping component -> (T, num_sensors) array
        """
        predictions = {}
        
        for component, data in sensor_data.items():
            if component == 'engine':
                model = self.engine_model
            elif component == 'transmission':
                model = self.transmission_model
            elif component == 'brake':
                model = self.brake_model
            elif component == 'battery':
                model = self.battery_model
            else:
                continue
            
            # Prepare input
            if len(data) < self.seq_length:
                # Pad with zeros
                pad = np.zeros((self.seq_length - len(data), data.shape[1]))
                data = np.concatenate([pad, data], axis=0)
            
            if len(data) > self.seq_length:
                data = data[-self.seq_length:]
            
            tensor = torch.FloatTensor(data).T.unsqueeze(0)  # (1, F, T)
            
            with torch.no_grad():
                rul = model(tensor).item()
            
            predictions[component] = {
                'rul_hours': rul,
                'rul_days': rul / 24,
                'maintenance_urgency': 'immediate' if rul < 24 else \
                                      'soon' if rul < 168 else \
                                      'normal' if rul < 720 else 'routine'
            }
        
        return predictions
    
    def predict_fleet_health(self, 
                              vehicle_data: Dict[str, Dict]) -> Dict:
        """
        Aggregate RUL predictions across entire fleet.
        """
        fleet_stats = {
            'total_vehicles': len(vehicle_data),
            'urgent_maintenance': 0,
            'soon_maintenance': 0,
            'average_rul': 0,
            'component_alerts': {}
        }
        
        total_rul = 0
        for vehicle_id, sensors in vehicle_data.items():
            pred = self.predict_rul(sensors)
            
            vehicle_min_rul = float('inf')
            for component, info in pred.items():
                fleet_stats['component_alerts'].setdefault(component, {
                    'urgent': 0, 'soon': 0, 'normal': 0
                })
                fleet_stats['component_alerts'][component][info['maintenance_urgency']] += 1
                vehicle_min_rul = min(vehicle_min_rul, info['rul_hours'])
            
            if vehicle_min_rul < 24:
                fleet_stats['urgent_maintenance'] += 1
            elif vehicle_min_rul < 168:
                fleet_stats['soon_maintenance'] += 1
            
            total_rul += vehicle_min_rul
        
        fleet_stats['average_rul'] = total_rul / len(vehicle_data) if vehicle_data else 0
        
        return fleet_stats
```

---

## Route Optimization & Logistics

### Vehicle Routing Problem with OR-Tools

```python
from typing import List, Dict, Tuple
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np

class RouteOptimizer:
    """
    AI-powered route optimizer for delivery fleets.
    Handles Vehicle Routing Problem with Time Windows (VRPTW),
    capacity constraints, and driver break scheduling.
    """
    
    def __init__(self, traffic_model=None):
        self.traffic_model = traffic_model  # ML model for travel time prediction
    
    def optimize(self, 
                 orders: List[Dict],
                 vehicles: List[Dict],
                 depot: Tuple[float, float]) -> Dict:
        """
        Solve VRPTW with capacity and time constraints.
        
        Args:
            orders: [{'id': str, 'lat': float, 'lon': float, 
                      'weight': float, 'time_window': (start, end), 
                      'service_time': int}]
            vehicles: [{'capacity': float, 'max_route_time': int}]
            depot: (lat, lon)
        
        Returns:
            Routes with ordered stops and ETAs
        """
        num_orders = len(orders)
        num_vehicles = len(vehicles)
        
        # Create data model
        data = self._create_data_model(orders, vehicles, depot)
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            num_orders + 1, num_vehicles, 0  # 0 = depot
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register transit callback
        def time_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return self._get_travel_time(data, from_node, to_node)
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time dimension
        routing.AddDimension(
            transit_callback_index,
            60,  # Max waiting time per stop (minutes)
            600,  # Max time per vehicle (minutes)
            True,  # Start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints
        for order_idx in range(1, num_orders + 1):
            start = data['time_windows'][order_idx][0]
            end = data['time_windows'][order_idx][1]
            time_dimension.CumulVar(order_idx).SetRange(start, end)
        
        # Add depot time window
        time_dimension.CumulVar(0).SetRange(0, 720)  # 12-hour operating window
        
        # Add capacity dimension
        def demand_callback(from_idx):
            from_node = manager.IndexToNode(from_idx)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # Null capacity slack
            data['vehicle_capacities'],
            True,  # Start cumul to zero
            'Capacity'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30
        search_parameters.log_search = True
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._format_solution(data, manager, routing, solution)
        else:
            return {'status': 'NO_SOLUTION_FOUND', 'routes': []}
    
    def _create_data_model(self, orders, vehicles, depot):
        """Create data model for OR-Tools."""
        data = {}
        
        # Locations: depot first, then orders
        data['locations'] = [(depot[0], depot[1])]
        for order in orders:
            data['locations'].append((order['lat'], order['lon']))
        
        # Time windows
        data['time_windows'] = [(0, 720)]  # Depot window
        for order in orders:
            data['time_windows'].append(order['time_window'])
        
        # Demands (depot has 0 demand)
        data['demands'] = [0]
        for order in orders:
            data['demands'].append(order['weight'])
        
        # Vehicle capacities
        data['vehicle_capacities'] = [v['capacity'] for v in vehicles]
        
        # Service times
        data['service_times'] = [0]  # Depot
        for order in orders:
            data['service_times'].append(order.get('service_time', 10))
        
        return data
    
    def _get_travel_time(self, data, from_node, to_node):
        """Get travel time between two nodes in minutes."""
        if from_node == to_node:
            return 0
        
        # Use ML model if available, otherwise use haversine + average speed
        lat1, lon1 = data['locations'][from_node]
        lat2, lon2 = data['locations'][to_node]
        
        # Haversine distance
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c  # km
        
        # Average speed 30 km/h for urban delivery
        travel_time = (distance / 30) * 60  # minutes
        
        # Add service time at origin
        travel_time += data['service_times'][from_node]
        
        return int(travel_time)
    
    def _format_solution(self, data, manager, routing, solution):
        """Format OR-Tools solution into readable route structure."""
        routes = []
        
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route_stops = []
            
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                
                stop = {
                    'stop_id': node_idx,
                    'arrival_time': solution.Value(time_var),
                    'location': data['locations'][node_idx] if node_idx < len(data['locations']) else None
                }
                route_stops.append(stop)
                
                index = solution.Value(routing.NextVar(index))
            
            if len(route_stops) > 1:  # More than just depot
                routes.append({
                    'vehicle_id': vehicle_id,
                    'stops': route_stops,
                    'total_stops': len(route_stops) - 1,
                    'total_time': route_stops[-1]['arrival_time']
                })
        
        return {
            'status': 'SOLUTION_FOUND',
            'total_distance': solution.ObjectiveValue(),
            'routes': routes
        }
```

### Reinforcement Learning for Delivery Routing

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class DeliveryRoutingEnv(gym.Env):
    """
    Gym environment for learning delivery routing policies.
    
    State: Current location, remaining packages, time, vehicle capacity
    Action: Next delivery stop (discrete)
    Reward: -distance_traveled - late_delivery_penalty + completion_bonus
    """
    
    def __init__(self, 
                 num_stops: int = 20,
                 max_episode_steps: int = 50):
        super().__init__()
        
        self.num_stops = num_stops
        self.max_steps = max_episode_steps
        
        # Observation: [current_x, current_y, time_remaining, 
        #                remaining_packages * 2 (x, y for each)]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(2 + 1 + num_stops * 2,),
            dtype=np.float32
        )
        
        # Action: which stop to visit next (0 = depot, 1..num_stops)
        self.action_space = spaces.Discrete(num_stops + 1)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Generate random stop locations
        self.stops = np.random.rand(self.num_stops, 2)
        self.depot = np.random.rand(2)
        
        # State
        self.current_pos = self.depot.copy()
        self.visited = np.zeros(self.num_stops, dtype=bool)
        self.time_elapsed = 0
        self.total_distance = 0
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        remaining = self.stops[~self.visited].ravel()
        if len(remaining) < self.num_stops * 2:
            padding = np.zeros(self.num_stops * 2 - len(remaining))
            remaining = np.concatenate([remaining, padding])
        
        return np.concatenate([
            self.current_pos,
            [self.max_steps - self.current_step],
            remaining
        ]).astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        if action == 0:  # Return to depot
            dist = np.linalg.norm(self.current_pos - self.depot)
            self.total_distance += dist
            self.current_pos = self.depot.copy()
            
            # Check if all delivered
            if self.visited.all():
                reward = 100 - self.total_distance  # Completion bonus
            else:
                reward = -dist - 50  # Penalty for returning early
        else:
            stop_idx = action - 1
            if self.visited[stop_idx]:
                reward = -10  # Penalty for re-visiting
            else:
                dest = self.stops[stop_idx]
                dist = np.linalg.norm(self.current_pos - dest)
                self.total_distance += dist
                self.current_pos = dest
                self.visited[stop_idx] = True
                
                # Time penalty proportional to distance
                reward = -dist
        
        terminated = self.visited.all() or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        pass


def train_delivery_rl_model():
    """Train a PPO agent for delivery routing."""
    env = DummyVecEnv([lambda: DeliveryRoutingEnv(num_stops=20)])
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    model.learn(total_timesteps=500000)
    model.save('delivery_routing_ppo')
    
    return model
```

---

## Drone Delivery Systems

### Autonomous Drone Navigation

```python
import torch
import torch.nn as nn
import numpy as np

class DroneNavigationPolicy(nn.Module):
    """
    End-to-end drone navigation policy for autonomous delivery.
    Processes depth images, GPS, and IMU data to output control commands.
    """
    
    def __init__(self, 
                 depth_shape: tuple = (1, 180, 320),
                 num_actions: int = 4):  # forward, left, right, hover
        super().__init__()
        
        # Depth CNN
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # State encoder (GPS + IMU)
        self.state_encoder = nn.Sequential(
            nn.Linear(7, 64),  # (lat, lon, alt, roll, pitch, yaw, speed)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Goal encoder (target GPS coordinates)
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 64),  # (target_lat, target_lon, target_alt)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Fusion
        fusion_dim = 128 + 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Action head (discrete)
        self.action_head = nn.Linear(256, num_actions)
        
        # Continuous control head (throttle, yaw_rate)
        self.control_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
    
    def forward(self, 
                depth: torch.Tensor,
                state: torch.Tensor,
                goal: torch.Tensor) -> dict:
        """
        Args:
            depth: (B, 1, H, W)
            state: (B, 7)
            goal: (B, 3)
        """
        depth_feat = self.depth_encoder(depth)
        state_feat = self.state_encoder(state)
        goal_feat = self.goal_encoder(goal)
        
        fused = torch.cat([depth_feat, state_feat, goal_feat], dim=1)
        features = self.fusion(fused)
        
        action_logits = self.action_head(features)
        continuous_control = self.control_head(features)
        
        return {
            'action_logits': action_logits,
            'throttle': continuous_control[:, 0],  # -1 to 1
            'yaw_rate': continuous_control[:, 1],  # -1 to 1
        }
```

---

## Case Studies

### Waymo Autonomous Driving Stack

Waymo (formerly the Google Self-Driving Car Project) operates the most mature autonomous driving system, with over 20 million miles of public road driving and billions of miles in simulation.

**Technical Architecture**:

1. **Sensor Suite**:
   - 5x LiDAR (1x long-range 360° + 4x short-range surround)
   - 29x cameras (360° coverage at multiple focal lengths)
   - 6x radar (forward, rear, side-facing)
   - GPS + IMU + wheel encoders

2. **Perception System**:
   - **Waymo Open Dataset**: 1,000+ driving scenes, 12M+ 3D labels
   - **BEV Transformer**: Fuses camera and LiDAR into unified bird's eye view representation
   - **PointPillars**: Efficient LiDAR feature extraction
   - **DETR3D**: Multi-camera 3D object detection
   - **Tracking**: Hungarian algorithm + Kalman filtering for multi-object tracking

3. **Prediction (Motion Forecasting)**:
   - **VectorNet**: Contextual scene representation using vectorized road graph
   - **TNT (Target-driven Trajectory Prediction)**: Goal-conditioned trajectory generation
   - **Scene Transformer**: End-to-end joint prediction for all agents

4. **Planning**:
   - **Hybrid A***: Path planning in complex environments
   - **MPC (Model Predictive Control)**: Trajectory tracking with kinematic constraints
   - **RL-based Contingency Planning**: Learned policies for edge-case handling

5. **Simulation**:
   - **Carcraft**: Waymo's simulation environment with 20,000+ parallel instances
   - **SurfelGAN**: Realistic LiDAR simulation using generative models
   - **Simulation metrics**: 15,000+ scenarios, 20M+ virtual miles driven daily

```python
class WaymoPerceptionPipeline:
    """
    Simplified representation of Waymo's perception pipeline.
    """
    
    def __init__(self):
        self.lidar_model = PointPillar()
        self.camera_model = BEVFormer(num_cameras=29)
        self.fusion = SensorFusionTransformer()
        self.tracker = self._init_tracker()
    
    def _init_tracker(self):
        """Multi-object tracker with Kalman filtering."""
        class MOTTracker:
            def __init__(self):
                self.tracks = {}
                self.next_id = 0
            
            def update(self, detections):
                # Hungarian algorithm matching
                # Kalman filter state update
                pass
        
        return MOTTracker()
    
    def process_frame(self, 
                       lidar_points: np.ndarray,
                       camera_images: list,
                       radar_data: list) -> dict:
        """Process one frame of sensor data."""
        
        lidar_feat = self.lidar_model(lidar_points)
        camera_feat = self.camera_model(camera_images)
        radar_feat = self._process_radar(radar_data)
        
        fused = self.fusion(camera_feat, lidar_feat, radar_feat)
        
        # Object detection heads
        detections = self._decode_detections(fused)
        
        # Tracking
        tracked_objects = self.tracker.update(detections)
        
        return {
            'detections': detections,
            'tracks': tracked_objects,
            'bev_features': fused
        }
```

### Tesla Full Self-Driving (FSD)

Tesla's approach differs fundamentally from Waymo — relying on cameras only (no LiDAR or radar in recent versions) and a massive neural network that processes raw video input end-to-end.

**Key Technical Details**:

1. **Sensor Suite**: 8 cameras (12 in Cybertruck), 3 front-facing (wide, main, narrow), 2 side-facing, 2 rear-facing. No LiDAR. No radar (removed in 2021 for Model 3/Y).

2. **Neural Network Architecture**:
   - **HydraNets**: Multi-task vision architecture shared across all Tesla vehicles
   - **Occupancy Network**: Learned volumetric occupancy grid replacing LiDAR
   - **BEV Network**: Multi-camera transformer for bird's eye view representation
   - **Objective**: Predicts 3D position, velocity, acceleration, and uncertainty

3. **Training**:
   - **4 Billion+ miles of fleet data**
   - **100,000+ GPU cluster (Dojo + NVIDIA)** for training
   - **Auto-labeling**: Automated 3D labeling using multi-camera reconstruction
   - **Shadow mode**: FSD decisions recorded but not acted upon for validation

4. **FSD Versions**:
   - **FSD v12**: "End-to-end" — single neural network from camera input to steering/throttle output, removing all hard-coded rules
   - **FSD v13**: Improved highway performance, 3x parameter count, 5Hz video input

```python
class TeslaOccupancyNetwork(nn.Module):
    """
    Simplified Tesla Occupancy Network — learned volumetric occupancy
    that replaces the need for LiDAR by estimating empty vs. occupied
    space from camera images.
    """
    
    def __init__(self, 
                 num_cameras: int = 8,
                 voxel_resolution: tuple = (200, 200, 16),  # x, y, z
                 voxel_range: tuple = (-50, 50, -50, 50, -2, 5)):
        super().__init__()
        
        self.voxel_res = voxel_resolution
        self.voxel_range = voxel_range
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ... EfficientNet or RegNet backbone
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512)
        )
        
        # Camera transformer to BEV
        self.cam_to_bev = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # 3D occupancy decoder (sparse convolutional or MLP-based)
        self.occupancy_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Occupancy probability
        )
    
    def forward(self, camera_images: list) -> torch.Tensor:
        """
        camera_images: list of 8 tensors (B, 3, H, W)
        
        Returns: (B, 200, 200, 16) occupancy grid
        """
        features = []
        for img in camera_images:
            feat = self.backbone(img)
            features.append(feat)
        
        # Fuse multi-camera features
        stacked = torch.stack(features, dim=1)  # (B, 8, 512)
        
        # Generate BEV queries (one per voxel)
        # ... (simplified)
        
        # Output occupancy grid
        occupancy = self.occupancy_decoder(stacked)
        
        return occupancy.view(-1, 200, 200, 16)
```

### Uber Freight & Logistics Optimization

Uber Freight uses AI to optimize freight matching, dynamic pricing, and route planning across a network of 50,000+ carriers and hundreds of shippers.

**Technical Capabilities**:

1. **Load-to-Carrier Matching**: ML model predicts optimal carrier for each load based on:
   - Carrier history and reliability score
   - Equipment type compatibility
   - Real-time location and empty miles
   - Driver hours-of-service compliance

2. **Dynamic Pricing**: Neural network for real-time freight rate optimization:
   - Market demand/supply elasticity model
   - Lane-specific rate baselines
   - Seasonal and event-based adjustments

3. **Route Optimization**: Multi-constraint VRP solver:
   - Pickup and delivery time windows
   - Driver break and rest schedules
   - Trailer type and weight restrictions
   - Toll road avoidance/optimization

4. **Predictive ETAs**: LSTM-based arrival time prediction:
   - Historical traffic patterns
   - Weather conditions
   - Construction and event data
   - Driver behavior models

```python
class UberFreightOptimizer:
    """
    Simplified Uber Freight load matching and pricing system.
    """
    
    def __init__(self):
        self.matching_model = self._build_matching_model()
        self.pricing_model = self._build_pricing_model()
    
    def _build_matching_model(self):
        """Gradient boosting model for load-to-carrier matching."""
        from lightgbm import LGBMRanker
        return LGBMRanker(
            objective='lambdarank',
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05
        )
    
    def _build_pricing_model(self):
        """Neural network for dynamic rate pricing."""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def match_loads(self, 
                     load: Dict,
                     available_carriers: List[Dict]) -> List[Dict]:
        """
        Rank carriers for a given load.
        """
        features = []
        for carrier in available_carriers:
            feat = self._extract_match_features(load, carrier)
            features.append(feat)
        
        scores = self.matching_model.predict(features)
        
        ranked = sorted(
            zip(available_carriers, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [{'carrier': c, 'score': s} for c, s in ranked]
    
    def predict_rate(self, 
                     origin: tuple, 
                     destination: tuple,
                     load_type: str,
                     weight: float,
                     market_conditions: Dict) -> float:
        """Predict optimal freight rate for a shipment."""
        features = np.array([
            origin[0], origin[1],  # lat, lon
            destination[0], destination[1],
            weight,
            market_conditions['supply_demand_ratio'],
            market_conditions['fuel_price'],
            market_conditions['avg_rate_per_mile'],
            len(market_conditions.get('available_carriers', []))
        ]).reshape(1, -1)
        
        with torch.no_grad():
            rate = self.pricing_model(torch.FloatTensor(features))
        
        return float(rate.item() * 100)  # Convert to dollars
```

### Amazon Prime Air Drone Delivery

Amazon's Prime Air drone delivery system uses AI for autonomous navigation, obstacle avoidance, and precision landing.

**Technical Architecture**:

1. **Airframe**: Hexacopter design, 40+ mph cruise speed, 5kg payload capacity
2. **Sensors**: Depth cameras, downward-facing LiDAR, GPS, IMU, barometer
3. **Navigation Stack**:
   - **Global Planner**: A* on pre-computed route corridors
   - **Local Planner**: Model Predictive Control with obstacle avoidance
   - **Landing Site Detection**: CNN for safe landing zone identification
4. **Airspace Management**:
   - **UTM (UAS Traffic Management)**: Integration with FAA's UTM system
   - **Geo-fencing**: Geospatial boundaries and no-fly zones
   - **Collision Avoidance**: ADS-B + ACAS-like conflict resolution

---

## Cross-References

This document intersects with several other domains in the AI Applications series:

- **[03-Finance-AI.md](03-Finance-AI.md)**: Algorithmic trading uses similar RL techniques (PPO, SAC) as delivery routing optimization. Risk management models in finance parallel safety assurances in autonomous driving.

- **[04-Manufacturing-AI.md](04-Manufacturing-AI.md)**: Predictive maintenance for manufacturing equipment uses identical autoencoder and RUL architectures as fleet vehicle maintenance. Digital twins for factories share concepts with autonomous driving simulation environments.

- **[02-Healthcare-AI.md](02-Healthcare-AI.md)**: Medical image segmentation (U-Net) shares architectural patterns with road scene segmentation. The regulatory validation frameworks for medical AI inform autonomous vehicle safety certification.

- **[07-Media-Entertainment-AI.md](07-Media-Entertainment-AI.md)**: Real-time graphics rendering and game AI share optimization techniques with autonomous driving simulators. GAN-based image generation (e.g., SurfelGAN) is used for synthetic LiDAR data generation.

- **[08-Agriculture-AI.md](08-Agriculture-AI.md)**: Agricultural drone navigation systems share path planning algorithms with drone delivery systems. Computer vision for plant detection uses the same YOLO/ResNet models as vehicle detection.

- **[10-Energy-AI.md](10-Energy-AI.md)**: EV fleet charging optimization intersects with smart grid load balancing. Battery health monitoring for EVs uses the same techniques as grid storage monitoring.

---

## Summary & Conclusion

This document has provided a deep technical exploration of AI applications across transportation and logistics, covering:

1. **Autonomous Vehicle Perception**: Multi-modal sensor fusion with cameras, LiDAR, and radar; BEV transformers (BEVFormer, DETR3D); occupancy networks; PointPillars and VoxelNet for LiDAR processing.

2. **Path Planning & Control**: Hybrid A* search, Reeds-Shepp curves, Model Predictive Control, and imitation learning (Conditional Imitation Learning) for end-to-end driving.

3. **Traffic Prediction**: Graph Neural Networks (GCN, DCRNN) for traffic flow forecasting on road networks; Diffusion Convolutional RNNs for state-of-the-art spatio-temporal prediction.

4. **Predictive Maintenance**: Remaining Useful Life (RUL) estimation for fleet vehicles using temporal convolutional networks, multi-component health monitoring.

5. **Route Optimization**: OR-Tools for Vehicle Routing Problem with Time Windows (VRPTW); Reinforcement Learning (PPO) for delivery routing; dynamic pricing models for freight.

6. **Drone Delivery**: End-to-end navigation policies combining depth vision, GPS, and IMU; airspace management; integration with UTM systems.

Key technical trends driving transportation AI:

- **End-to-End Learning**: Moving from modular perception-planning-control pipelines to single neural networks (e.g., Tesla FSD v12)
- **BEV-Centric Architecture**: Bird's eye view transformers as the unified representation for sensor fusion
- **Simulation-First Development**: Extensive use of photorealistic simulation for training and validation
- **Predictive Safety Models**: Learned occupancy and uncertainty quantification replacing hard-coded safety rules
- **Edge Computing**: On-vehicle inference optimization using TensorRT, quantization, and hardware acceleration

The path to widespread autonomous transportation deployment requires continued advances in: causal reasoning for rare event handling, validation and certification frameworks for safety-critical AI, computational efficiency for real-time edge inference, and coordination systems across connected vehicles and infrastructure.
