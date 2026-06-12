# AI for Robotics: Perception, Planning, and Control

## Table of Contents
1. [Introduction](#1-introduction)
2. [Robot Perception](#2-robot-perception)
3. [Motion Planning](#3-motion-planning)
4. [Control](#4-control)
5. [Robot Learning](#5-robot-learning)
6. [Sim-to-Real Transfer](#6-sim-to-real-transfer)
7. [Vision-Language-Action (VLA) Models](#7-vision-language-action-vla-models)
8. [Simulation](#8-simulation)
9. [Hardware Platforms](#9-hardware-platforms)
10. [Practical Guidance: Choosing Approaches](#10-practical-guidance-choosing-approaches)
11. [Cross-References](#11-cross-references)

---

## 1. Introduction

Robotics AI combines computer vision, reinforcement learning, control theory, and simulation to create robots that perceive, plan, and act in the physical world. Recent advances in deep learning have dramatically improved robot capabilities in manipulation, locomotion, and navigation.

### The Perception–Planning–Control Loop

Most robotic systems follow a three-stage pipeline:

```
[ Sensors ] → Perception → State Estimation → Planning → Control → [ Actuators ]
                                 ↑                                              │
                                 └──────────── Feedback Loop ──────────────────┘
```

- **Perception** extracts meaning from raw sensor data (cameras, lidar, tactile sensors, force/torque).
- **Planning** generates trajectories or action sequences to achieve goals while avoiding obstacles.
- **Control** converts planned trajectories into low-level motor commands, closing the loop with feedback.

Modern AI has blurred these boundaries: end-to-end learned policies (e.g., VLA models) can bypass explicit planning and control stages entirely.

---

## 2. Robot Perception

### Object Detection

Real-time object detection is the backbone of pick-and-place, navigation, and human-robot interaction.

| Model | Type | Speed | mAP | Robotics Use Case |
|-------|:----:|:----:|:---:|-------------------|
| **YOLOv8/v9/v10** | One-stage CNN | 100–300+ FPS | 50–55% | Real-time grasping, obstacle avoidance |
| **DETR** | Transformer | 10–30 FPS | 45–50% | Small-batch, high-accuracy picking |
| **ViT (DeiT, DINOv2)** | Vision Transformer | 20–60 FPS | 52–58% | Open-vocabulary detection in cluttered scenes |
| **Grounding DINO** | Open-vocabulary | 10–30 FPS | 56% | Zero-shot detection from language prompts |

```python
# Example: Real-time object detection with YOLOv8 for robot pick-and-place
from ultralytics import YOLO
import cv2
import numpy as np

# Load pretrained model
model = YOLO("yolov8n-pose.pt")  # or yolov8n.pt for detection

# Camera input (simulated with a frame)
cap = cv2.VideoCapture(0)  # Replace with robot wrist camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    # Extract bounding boxes and classes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf > 0.5:
            # Compute grasp point as bounding box center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send (cx, cy, cls) to motion planner
            # robot.grasp_at(cx, cy, cls)

    cv2.imshow("Robot Perception", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

### Depth Estimation

For 3D understanding — bin picking, collision checking, and scene reconstruction.

| Model | Input | Output | Inference | Strengths |
|-------|:----:|:------:|:---------:|-----------|
| **MiDaS v3.1** | Single RGB | Metric depth | ~30 FPS on GPU | Robust to diverse scenes |
| **Depth Anything V2** | Single RGB | Relative depth | ~40 FPS on GPU | Best zero-shot generalization |
| **ZoeDepth** | Single RGB | Metric depth | ~25 FPS | Metric depth without calibration |
| **Stereo (RAFT-Stereo)** | Stereo pair | Disparity map | ~15 FPS | High accuracy at close range |

```python
# Example: Depth estimation for collision-free grasping
import torch
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder="vitl", features=256, out_channels=256)
model.load_state_dict(torch.load("depth_anything_v2_vitl.pth",
                                 map_location="cuda"))
model.eval().cuda()

# --- Run on camera frame ---
with torch.no_grad():
    depth = model.infer_image(frame)  # HxW numpy array, normalized [0,1]

# Threshold: keep only pixels closer than 0.5m in depth space
grasp_mask = depth > 0.4  # Experimentally determined threshold
safe_pixel = np.median(np.argwhere(grasp_mask), axis=0)  # Safe approach vector
```

### Scene Understanding & Segmentation

| Model | Task | Output | Notes |
|-------|:----:|:------:|-------|
| **SAM 2 (Meta)** | Promptable segmentation | Binary masks per prompt | Interactive — click or box prompt |
| **SAM + Grounding DINO** | Open-vocabulary segmentation | Class-agnostic masks + labels | "Grasp the red mug" pipeline |
| **OpenScene** | 3D open-vocabulary | 3D point labels | For full 3D scene graphs |
| **OneFormer / Mask2Former** | Panoptic segmentation | Stuff + things | Unified indoor mapping |

```python
# Example: Open-vocabulary segmentation pipeline for "pick up the screwdriver"
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from PIL import Image
import torch

# Grounding DINO for object detection from text prompt
gdino = load_model("config/GroundingDINO_SwinT_OGC.py",
                   "weights/groundingdino_swint_ogc.pth")

image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
boxes, logits, phrases = predict(
    model=gdino,
    image=image_pil,
    caption="screwdriver · pliers · bolt",
    box_threshold=0.35,
    text_threshold=0.25,
    device="cuda",
)
# boxes can be passed to SAM 2 for pixel-perfect segmentation,
# then the mask centre becomes the grasp point.
```

### Tactile Sensing

| Sensor | Type | Resolution | Bandwidth | Price |
|--------|:----:|:----------:|:---------:|:-----:|
| **GelSight Mini** | Gel-based optical | 640×480 | 30 Hz | $3K |
| **DIGIT (Meta)** | Gel-based optical | 240×240 | 60 Hz | ~$300 |
| **BioTac SP** | Hydro-acoustic | Multi-modal | 100 Hz | $4K |
| **SynTouch BioTac** | Tri-modal (force, vibration, temp) | 19 electrodes | 100 Hz | $5K |

Tactile sensing is critical for:
- **Delicate grasping** (eggs, fruit, cables)
- **In-hand manipulation** (rotating a screwdriver without dropping)
- **Material classification** (slip detection, texture recognition)

```python
# Example: Slip detection from tactile images (simplified)
import torch.nn as nn

class SlipDetector(nn.Module):
    """ConvNet that takes a DIGIT/GelSight image and outputs slip probability."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cnn(x)  # → slip probability in [0, 1]

# Inference: if slip > 0.6: increase grip force
slip_net = SlipDetector().cuda()
slip_net.load_state_dict(torch.load("slip_detector.pth"))
slip_prob = slip_net(tactile_image.unsqueeze(0)).item()
```

---

## 3. Motion Planning

### Planner Comparison

| Planner | Completeness | Dimension | Speed | Smoothness | Use Case |
|---------|:-----------:|:---------:|:-----:|:----------:|----------|
| **RRT** | Probabilistic | Any | Fast | Low | High-DoF arms, exploration |
| **RRT\*** | Asymptotically optimal | Any | Medium | Medium | Near-optimal paths |
| **A\*** | Complete | 2–4 | Fast | N/A | Grid maps, navigation |
| **D\* Lite** | Complete (dynamic) | 2–4 | Medium | N/A | Dynamic environments |
| **CHOMP** | Gradient-based | Any | Fast | High | Smooth collision-free trajectories |
| **TrajOpt** | Sequential convex | Any | Medium | High | Industrial manipulation |
| **STOMP** | Stochastic | Any | Slow | High | High-DoF, constrained tasks |
| **MPNet** | Learned (neural) | Any | Very fast | Medium | Learned priors for planning |

### Code Example: RRT Path Planning

```python
import numpy as np
import random
import matplotlib.pyplot as plt

class RRT:
    """Rapidly-exploring Random Tree for a 2D point robot."""
    def __init__(self, start, goal, obstacle_list, bounds,
                 step_size=0.5, max_iter=1000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacle_list       # list of (center, radius)
        self.bounds = bounds                 # (x_min, x_max, y_min, y_max)
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.parent = [None]                 # parent index for each node

    def plan(self):
        for i in range(self.max_iter):
            # Sample random point (with 5% goal bias)
            if random.random() < 0.05:
                rand_pt = self.goal
            else:
                rand_pt = np.array([
                    random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3]),
                ])

            # Find nearest node
            dists = np.linalg.norm(np.array(self.nodes) - rand_pt, axis=1)
            nearest_idx = np.argmin(dists)
            nearest = self.nodes[nearest_idx]

            # Steer toward random point
            direction = rand_pt - nearest
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            step = direction / norm * min(self.step_size, norm)
            new_node = nearest + step

            # Collision check
            if self._collision_free(nearest, new_node):
                self.nodes.append(new_node)
                self.parent.append(nearest_idx)

                # Check if we reached the goal
                if np.linalg.norm(new_node - self.goal) < self.step_size:
                    return self._extract_path(len(self.nodes) - 1)
        return None  # No path found

    def _collision_free(self, a, b):
        """Check straight-line segment a→b against all obstacles."""
        n = int(np.linalg.norm(b - a) / 0.05) + 1
        for t in np.linspace(0, 1, n):
            pt = a + t * (b - a)
            for center, radius in self.obstacles:
                if np.linalg.norm(pt - center) <= radius:
                    return False
        return True

    def _extract_path(self, idx):
        path = []
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parent[idx]
        return path[::-1]


# --- Usage ---
if __name__ == "__main__":
    obstacles = [((0.5, 0.5), 0.2), ((1.5, 1.5), 0.3)]
    rrt = RRT(start=(0, 0), goal=(2, 2), obstacle_list=obstacles,
              bounds=(0, 3, 0, 3))
    path = rrt.plan()
    if path:
        print(f"Path found with {len(path)} waypoints")
        # path → send to trajectory smoother → controller
```

### Code Example: A\* Grid Planning

```python
import heapq

def astar(grid, start, goal):
    """A* search on a binary occupancy grid (0=free, 1=obstacle)."""
    h = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan
    rows, cols = grid.shape
    open_set = [(0 + h(start, goal), 0, start, None)]
    closed = {}
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)

        if current in closed:
            continue
        closed[current] = True
        came_from[current] = parent

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 1:  # obstacle
                continue
            ng = cost + 1
            neighbor = (nx, ny)
            if neighbor not in g_score or ng < g_score[neighbor]:
                g_score[neighbor] = ng
                f = ng + h(neighbor, goal)
                heapq.heappush(open_set, (f, ng, neighbor, current))
    return None

# --- Example ---
grid = np.zeros((10, 10), dtype=int)
grid[3:7, 3:7] = 1  # obstacle block
path = astar(grid, (0, 0), (9, 9))
if path:
    print(f"A* path length: {len(path)}")  # e.g., 18 steps
```

### Trajectory Optimization with CHOMP

CHOMP (Covariant Hamiltonian Optimization for Motion Planning) treats trajectory generation as an optimization problem:

```
minimize: F_total = F_obstacle + λ * F_smoothness
```

- **F_smoothness**: penalizes acceleration (sum of squared velocity changes)
- **F_obstacle**: a differentiable cost function based on signed-distance field of obstacles

```python
# Simplified CHOMP-like optimization (1D example)
import autograd.numpy as np
from autograd import grad

def chomp_optimize(init_traj, obstacle_cost_fn, steps=50, lr=0.1, lam=1.0):
    traj = init_traj.copy()
    n = len(traj)

    def smoothness(t):
        return np.sum((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2)

    def total_cost(t):
        return obstacle_cost_fn(t) + lam * smoothness(t)

    grad_fn = grad(total_cost)
    for _ in range(steps):
        traj -= lr * grad_fn(traj)
    return traj
```

Practical guidance: RRT is the go-to for high-DoF exploration, A\* for known 2D grids, and TrajOpt/CHOMP for smooth industrial trajectories. For learned planning, consider **MPNet** or **Neural Motion Planning** (Va3r et al.) when you have a dataset of solved planning problems.

---

## 4. Control

### Method Comparison (expanded)

| Method | Type | Model Required | Tuning Effort | Robustness | Best For |
|--------|:----:|:--------------:|:-------------:|:----------:|----------|
| **PID** | Classical | None | Low (3 gains) | Moderate | Fan speed, motor current, simple position |
| **LQR** | Optimal | Linear dynamics | Low (Q, R weights) | Low | Quadrotor hover, cart-pole, balancing |
| **MPC** | Model-based | Dynamics model | Medium (horizon, weights) | High | Quadrotor acro, autonomous driving, walking |
| **Impedance / Admittance** | Force-based | None (stiffness/damping) | Medium | High | Peg-in-hole, assembly, human-robot collaboration |
| **Passivity-based** | Energy-based | System model | Medium | Very high | Bilateral teleoperation, haptics |
| **RL (PPO, SAC)** | Data-driven | None | High (reward design) | Variable | Contact-rich dexterous manipulation |
| **IL (BC, IRL)** | Imitation | None | Medium (demo quality) | Low | Learning from human demonstration |

### PID Controller

```python
class PID:
    """Discrete-time PID controller with anti-windup."""
    def __init__(self, kp, ki, kd, dt=0.01, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt

        # Anti-windup clamp
        if self.output_limits:
            self.integral = np.clip(self.integral, *self.output_limits)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
        return output


# --- Joint position control example ---
pid = PID(kp=2.0, ki=0.5, kd=0.1, dt=0.01, output_limits=(-12.0, 12.0))

for t in range(1000):
    joint_pos = robot.read_joint_position()   # measurement
    torque = pid.compute(setpoint=np.pi / 4,  # target 45°
                         measurement=joint_pos)
    robot.set_joint_torque(torque)
```

### Model Predictive Control (MPC)

MPC solves a finite-horizon optimal control problem at each timestep, applying only the first action and receding.

```python
import cvxpy as cp
import numpy as np

class LinearMPC:
    """MPC for a linear system x_{k+1} = A x_k + B u_k."""
    def __init__(self, A, B, Q, R, N=10):
        self.A, self.B = A, B
        self.Q, self.R = Q, R
        self.N = N

    def solve(self, x0, x_ref=np.zeros(4)):
        n, m = self.B.shape
        x = cp.Variable((n, self.N + 1))
        u = cp.Variable((m, self.N))

        cost = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.N):
            cost += cp.quad_form(x[:, k] - x_ref, self.Q) + cp.quad_form(u[:, k], self.R)
            constraints.append(x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k])
            # Input bounds
            constraints.append(cp.abs(u[:, k]) <= 10.0)

        cost += cp.quad_form(x[:, self.N] - x_ref, self.Q)  # terminal cost

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        return u[:, 0].value  # return first control input

# --- Usage: quadrotor hover (simplified double integrator) ---
A = np.array([[1, 0, 0.01, 0],
              [0, 1, 0, 0.01],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0, 0], [0, 0], [0.01, 0], [0, 0.01]])
Q, R = np.eye(4) * 10, np.eye(2) * 0.1
mpc = LinearMPC(A, B, Q, R, N=15)

# At each control loop:
# u = mpc.solve(current_state, desired_xy+velocity)
```

### Impedance Control

Impedance control regulates the dynamic relationship between position and force, making the robot behave like a mass-spring-damper system.

```python
class ImpedanceController:
    """
    Cartesian impedance control:
    F = M * x_ddot_des + D * (x_dot_des - x_dot) + K * (x_des - x)
    """
    def __init__(self, M=1.0, D=20.0, K=100.0):
        self.M, self.D, self.K = M, D, K

    def compute_force(self, x_des, x_dot_des, x_cur, x_dot_cur):
        pos_err = x_des - x_cur
        vel_err = x_dot_des - x_dot_cur
        # Desired acceleration (set to 0 for quasi-static)
        x_ddot_des = np.zeros_like(x_cur)
        return self.M @ x_ddot_des + self.D @ vel_err + self.K @ pos_err

    def wrench_to_joint_torques(self, J, F_cart):
        """Convert Cartesian force to joint torques: τ = J^T F"""
        return J.T @ F_cart
```

### Practical Guidance on Control Selection

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Drone hover / balance | LQR | Linearized model near equilibrium is sufficient |
| Autonomous car path-following | MPC | Handles constraints, preview capability |
| Robot arm peg-in-hole | Impedance | Converts positional error to compliant force |
| High-speed pick-and-place | PID + feedforward | Simple, fast loop rate |
| Dexterous hand manipulation | RL (PPO/SAC) | Contact-rich, hard to model analytically |
| Bilateral teleoperation | Passivity-based | Stability guarantee under time delay |

---

## 5. Robot Learning

### Method Comparison (expanded)

| Method | Approach | Data Required | Sample Efficiency | Generalization | Key Challenge |
|--------|----------|:------------:|:-----------------:|:--------------:|---------------|
| **Behavioral Cloning (BC)** | Supervised imitation | Expert demos (100–10K) | High | Low | Covariate shift |
| **DAgger** | Interactive imitation | Demos + online corrections | Medium | Medium | Human-in-loop cost |
| **PPO / SAC (RL)** | Trial and error in sim | 1M–100M steps | Low | Medium | Reward design, sim gap |
| **Sim-to-Real** | Train in sim, deploy on real | Sim data + domain randomize | Medium | High | Reality gap |
| **Offline RL (CQL, IQL)** | Learn from static dataset | Pre-collected experience | N/A (offline) | Medium | Distribution shift |
| **IL + RL hybrid (GAIL, AIRL)** | Imitation + RL | Demos + environment | Medium | Medium | Mode collapse |
| **VLA (RT-2, Octo, π0)** | Foundation model | Massive multi-robot + web | Zero-shot | Very high | Compute, latency, fine-tuning |

### Behavioral Cloning

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Policy: image → joint angles
class BCPolicy(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.backbone = torch.hub.load("pytorch/vision", "resnet18",
                                        weights="ResNet18_Weights.IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(512, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, img):
        feat = torch.relu(self.backbone(img))
        return self.head(feat)  # predicted joint angles

# Training loop
policy = BCPolicy().cuda()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(50):
    for imgs, actions in dataloader:   # expert demonstrations
        pred = policy(imgs.cuda())
        loss = loss_fn(pred, actions.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### RL with PPO for Continuous Control

```python
# Pseudocode outline — full PPO implementations use `stable-baselines3`
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create environment (e.g., reaching task in MuJoCo)
env = make_vec_env("FetchReach-v2", n_envs=4)

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
model.save("fetch_reach_ppo")

# Deploy
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

### Offline RL

Offline RL learns entirely from pre-collected datasets without environment interaction — ideal when the real robot is too valuable/slow to collect online data.

```python
# Using d3rlpy — popular offline RL library
# pip install d3rlpy
import d3rlpy

dataset = d3rlpy.dataset.MDPDataset.create_from_arrays(
    observations=observations_np,  # (N, obs_dim)
    actions=actions_np,            # (N, action_dim)
    rewards=rewards_np,            # (N,)
    terminals=terminals_np,        # (N,) — bool
)

cql = d3rlpy.algos.CQL(
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    batch_size=256,
    n_critics=2,
)
cql.fit(dataset, n_steps=500_000)
cql.save_model("offline_policy.d3")
```

### Key Systems (expanded)

| System | Organization | Backbone | Training Data | Capabilities | Availability |
|--------|:-----------:|:--------:|:------------:|:------------:|:------------:|
| **RT-2** | Google DeepMind | PaLM-E | Robot + web data | Generalist manipulation | Proprietary |
| **RT-2-X** | Google DeepMind | RT-2 + cross-embodiment | Multi-robot data | Zero-shot transfer | Proprietary |
| **Octo** | UC Berkeley | Transformer | Open X-Embodiment | Language-conditioned | Open-source |
| **π0 (Physical Intelligence)** | Physical Intelligence | Vision-Language Model | Wide robot data | Generalist policy | Private beta |
| **GROOT (Nvidia)** | Nvidia | Diffusion policy | Egocentric video | Humanoid loco-manipulation | Proprietary |
| **DROID** | Multi-institute | Dataset + offline RL | 350+ robots, 1.5 years | Distributed scale | Open dataset |
| **OpenVLA** | Stanford/CMU | Prismatic-VLM | Open X-Embodiment | Open-source VLA | Apache 2.0 |

---

## 6. Sim-to-Real Transfer

Sim-to-real is the dominant paradigm for scaling robot learning. The core challenge is the **reality gap** — the mismatch between simulation and the real world.

### Domain Randomization

Randomize simulation parameters during training so the policy generalizes to real-world variation:

```python
# Example: Domain randomization in Isaac Sim / MuJoCo
domain_randomization = {
    "physics": {
        "friction": np.random.uniform(0.2, 2.0),
        "damping": np.random.uniform(0.5, 1.5),
        "mass_scale": np.random.uniform(0.8, 1.2),
    },
    "rendering": {
        "lighting": np.random.choice(["bright", "dim", "side"]),
        "texture": np.random.choice(["plain", "noise", "checkerboard"]),
        "camera_noise": np.random.uniform(0.0, 0.02),
    },
    "actuator": {
        "latency": np.random.uniform(0.0, 0.05),   # seconds
        "torque_noise": np.random.uniform(0.0, 0.1),  # fraction
    },
    "initial_state": {
        "joint_offset": np.random.uniform(-0.1, 0.1, n_joints),
    },
}
```

### System Identification (SysID)

Fit simulation parameters to match real-robot data before training:

```python
# Minimize discrepancy between real and simulated trajectories
def sysid_loss(params, real_trajectories):
    """
    params: [mass, friction, damping, actuator_gains...]
    real_trajectories: list of joint position/velocity sequences
    """
    sim_trajectories = simulate(params)  # run simulator with these params
    mse = np.mean((real_trajectories - sim_trajectories) ** 2)
    return mse

# scipy.optimize.minimize(sysid_loss, x0=nominal_params,
#                          args=(real_traj,), method="Nelder-Mead")
```

### Sim-to-Real Pipeline

```python
# High-level pipeline pseudocode
def sim_to_real_pipeline():
    # Step 1: System identification on real data
    params = identify_system_parameters(real_robot_logs)

    # Step 2: Train in simulation with domain randomization
    env = RandomizedSimEnv(params, randomize_kwargs)
    policy = PPO("MlpPolicy", env).learn(5_000_000)

    # Step 3: Zero-shot deploy
    policy_deploy = policy.actor  # strip training noise
    deploy_on_real_robot(policy_deploy)

    # Step 4: (Optional) Fine-tune with small real-world dataset
    real_env = RealRobotEnv()
    policy = PPO("MlpPolicy", real_env).learn(50_000)  # ~10 minutes
```

### Sim-to-Real Success Map

| Task | Sim Method | Real Result | Gap-Closing Technique |
|------|:----------:|:-----------:|:---------------------:|
| **In-hand cube rotation** | PPO in MuJoCo | Deployed to Shadow Hand | Domain randomization + random friction |
| **Grasping diverse objects** | RL in Isaac Gym | Zero-shot on Franka | Random cameras, textures, masses |
| **Quadrotor acrobatics** | MPC in simulation | 90% success real flight | SysID + disturbance estimation |
| **Door opening** | SAC in PyBullet | Franka Panda | Random handle pose + force limits |
| **Bimanual assembly** | RL in Isaac Sim | Universal Robots | Curriculum: sim → sim w/ noise → real |

---

## 7. Vision-Language-Action (VLA) Models

VLAs are the latest frontier — large models that take vision + language input and directly output robot actions.

### Architecture (simplified)

```
[Image tokens] ─┐
                ├─→ Pretrained VLM (e.g., PaLI, Prismatic) → Action head → Joint angles / EE pose
[Text tokens]  ─┘
```

### How They Work

1. **Visual encoder** (ViT) encodes camera images into token embeddings.
2. **Text encoder** encodes the instruction (e.g., "pick up the blue mug").
3. **LLM backbone** (PaLM, LLaMA, or specialized transformer) fuses visual and text tokens.
4. **Action head** — either discrete bins (RT-2: 256 bins per dimension) or continuous regression (π0: diffusion head) — outputs motor commands.

### Comparison of VLA Models

| Model | LLM Backbone | Action Representation | Max Horizon | Latency | Open-Source |
|-------|:------------:|:---------------------:|:-----------:|:-------:|:-----------:|
| **RT-2** | PaLM-E (540B) | Discrete bins | 1 step | ~1 s | No |
| **RT-2-X** | PaLM-E (540B) | Discrete bins | 1 step | ~1 s | No |
| **Octo** | Small Transformer | Continuous (diffusion) | 20 steps | ~200 ms | Yes (Apache 2.0) |
| **OpenVLA** | Prismatic-7B | Discrete bins | 1 step | ~500 ms | Yes (Apache 2.0) |
| **π0** | Custom DiT | Flow-matching diffusion | 50 steps | ~100 ms | No |
| **GROOT** | Custom ViT + DiT | Diffusion | 16 steps | ~30 ms | No |

### Example: Loading and Running OpenVLA

```python
# pip install openvla
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

# Prepare input
image = Image.open("robot_view.jpg")
prompt = "pick up the red cube"

inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

# Generate action
with torch.no_grad():
    action = model.predict_action(**inputs, unnorm_key="bridge_dataset")
    # action shape: (7,) — [x, y, z, roll, pitch, yaw, gripper]
    print(f"Predicted EE pose: {action}")

# Send to robot
# robot.move_to_pose(action[:3], action[3:6])
# robot.gripper(action[6])
```

### When to Use VLA vs. Traditional RL/Planning

| Criterion | Use VLA | Use Traditional RL | Use Classical Planning + Control |
|-----------|:-------:|:------------------:|:-------------------------------:|
| **Data available** | Large diverse datasets | Simulation budget | Domain knowledge available |
| **Task specificity** | Few-shot generalist | Single-task specialist | Well-understood task |
| **Compute at inference** | GPU needed (500 ms+) | CPU possible | Real-time (kHz rates) |
| **Safety requirements** | Black-box (hard to certify) | Somewhat interpretable | Provable guarantees |
| **Open-vocabulary** | Yes | No | No |

---

## 8. Simulation

### Simulator Comparison (expanded)

| Simulator | Physics Engine | Rendering | Native Language | License | Best For |
|-----------|:-------------:|:---------:|:---------------:|:-------:|----------|
| **MuJoCo** | Custom (convex approx.) | OpenGL (minimal) | Python, C, C++ | Apache 2.0 | RL research, biomechanics |
| **Isaac Sim** | PhysX 5 | RTX (photorealistic) | Python | Free (NVIDIA) | Industrial manipulation, digital twins |
| **Isaac Gym** | PhysX 5 | None (headless) | Python | Free (NVIDIA) | RL at scale (10K envs) |
| **SAPIEN** | PhysX 5 | Good (VisionBlazer) | Python | MIT | Manipulation, part-level assembly |
| **PyBullet / Bullet3** | Bullet | Basic OpenGL | Python, C++ | Zlib | Education, rapid prototyping |
| **Habitat 2.0** | Bullet | Photorealistic (RedWood) | Python | MIT | Embodied navigation, rearrangement |
| **Drake** | Custom (accurate) | None | Python, C++, MATLAB | BSD-2-Clause | Control theory, rigid-body dynamics |
| **Genesis** | Custom GPU-accelerated | Photorealistic | Python | MIT | High-speed simulation (10K FPS) |
| **ManiSkill** | SAPIEN | Good | Python | MIT | Standardized manipulation benchmarks |

### Code Example: MuJoCo Setup

```python
# pip install mujoco
import mujoco

# Load model from XML
xml = """
<mujoco>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="robot" pos="0 0 0.2">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.1" rgba="0 0.5 1 1"/>
    </body>
    <body name="target" pos="0.5 0.5 0.2">
      <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Run simulation
for t in range(1000):
    data.ctrl[:] = [0.1, -0.05]  # control signal (force on joints)
    mujoco.mj_step(model, data)
    if t % 100 == 0:
        print(f"t={t}, robot pos: {data.qpos[:2]}")
```

### Code Example: Isaac Sim (via omniverse)

```python
# Isaac Sim scripting (from NVIDIA docs — runs inside Omniverse)
from omni.isaac.kit import SimulationApp
sim = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add a cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube",
        name="cube",
        position=np.array([0.0, 0.0, 0.5]),
        scale=np.array([0.1, 0.1, 0.1]),
        color=np.array([1.0, 0.0, 0.0]),
    )
)

# Physics stepping
for step in range(100):
    world.step(render=False)
    pos, _ = cube.get_world_pose()
    if step % 20 == 0:
        print(f"Step {step}: cube height = {pos[2]:.4f}")

sim.close()
```

### Simulation Benchmarks

| Benchmark | Simulator | Tasks | Metric | # Envs |
|-----------|:---------:|:-----:|:------:|:------:|
| **DM Control** | MuJoCo | 30+ continuous control | Return | 1 |
| **MetaWorld** | MuJoCo | 50 manipulation tasks | Success rate | 1 |
| **Isaac Gym / Orbit** | PhysX 5 | Dexterous manipulation, locomotion | Return / FPS | 10K+ |
| **Habitat 2.0** | Bullet | Navigation, rearrangement | SPL | 1–100 |
| **ManiSkill 3** | SAPIEN | 200+ assembly & manipulation | Success rate | 1–512 |
| **Surreal** | MuJoCo | Distributed RL | Return | 1K+ |

---

## 9. Hardware Platforms

### Platform Comparison (expanded)

| Platform | Type | DoF | Compute | Sensors (Built-in) | Max Payload | Cost | Best Use |
|----------|:----:|:---:|:-------:|:------------------:|:-----------:|:----:|----------|
| **Boston Dynamics Spot** | Quadruped | 12 | Edge AI (Intel) | 5x stereo, IMU, joint encoders | 14 kg | $75K | Inspection, construction, research |
| **Universal Robots UR5e** | Arm | 6 | External | Joint torque, force-torque (optional) | 5 kg | $35K | Light assembly, lab automation |
| **UR20** | Arm | 6 | External | Same | 20 kg | $40K | Heavier payload than UR5e |
| **Franka Emika Panda** | Arm | 7 | External | Joint torque (7-axis), cameras | 3 kg | $12K | Research, force-sensitive tasks |
| **Hello Robot Stretch 3** | Mobile manipulator | 8 | Onboard (Intel NUC) | Depth camera, wrist camera, IMU | 2 kg | $25K | Home robotics research |
| **Kuka LBR iiwa 14** | Industrial arm | 7 | External | Joint torque sensors | 14 kg | $50K+ | Industrial assembly, medical |
| **Fetch Robotics** | Mobile manipulator | 8 | Onboard | Lidar, RGB-D, microphone | 6 kg | $30K | Warehouse, logistics |
| **NVIDIA Jetson Orin** | Edge compute | N/A | 275 TOPS GPU | None (add sensors) | N/A | $2K | On-robot AI inference |
| **Anymal D (ANYbotics)** | Quadruped | 12 | Onboard | Lidar, stereo, IMU | 10 kg | $150K | Industrial inspection |
| **AgileX Limo** | Differential/omni | 2–4 | Raspberry Pi / Jetson | IMU, encoder, optional lidar | 4 kg | $2–5K | Education, SLAM research |
| **Aloha (Mobile Aloha)** | Bimanual mobile manipulator | 14 | Onboard (2× NUC) | Wrist cameras, force sensors | 1 kg/arm | $10K | Bimanual imitation learning |

### Compute Comparison for On-Device AI

| Module | AI Performance (INT8 TOPS) | Power | RAM | Typical Use | Price |
|--------|:--------------------------:|:-----:|:---:|:------------|:-----:|
| NVIDIA Jetson Orin Nano | 40 | 7–15 W | 8 GB | SLAM, lightweight detection | $250 |
| NVIDIA Jetson Orin NX | 100 | 10–25 W | 16 GB | Real-time YOLO, depth | $600 |
| NVIDIA Jetson AGX Orin | 275 | 15–60 W | 64 GB | On-policy RL, VLA inference | $2,000 |
| Intel RealSense + host CPU | — | 5 W (cam only) | — | Perception only | $300 |
| Google Coral TPU | 4 | 2 W | — | Edge ML accelerators | $60 |
| Hailo-8L | 13 | 2.5 W | — | Lightweight NN acceleration | $45 |

---

## 10. Practical Guidance: Choosing Approaches

### Task → Recommended Stack

| Task | Perception | Planning | Control | Learning Method |
|------|:----------:|:--------:|:-------:|:---------------:|
| **Pick & place known object** | YOLO / template | A* or RRT + smooth | PID / impedance | BC from 100 demos |
| **Pick arbitrary novel object** | Grounding DINO + SAM | Learned grasp sampler | Impedance | Sim-to-real with domain randomization |
| **Drone agile flight** | Depth Anything | MPC | LQR inner loop | None (hand-tuned) |
| **Legged locomotion** | Height map from depth | Foot placement planner | MPC + whole-body | RL in Isaac Gym |
| **Assembly (peg-in-hole)** | Vision + F/T sensing | TrajOpt | Admittance control | DAgger or offline RL |
| **Cloth folding** | SAM + keypoint detector | None (learned) | Impedance | BC + online RL fine-tune |
| **Bimanual dishwashing** | Open-vocabulary det. | Learned motion primitives | Impedance (arms) | VLA (π0, OpenVLA) |
| **Human-robot collaboration** | 3D pose estimation | RRT + safe distance | Passivity-based | None (model-based) |

### Decision Tree

```
What is the task?
├── Well-defined geometry & known dynamics
│   └── Classical: CAD-based grasping → RRT → PID
├── Repetitive with many examples available
│   └── Imitation: Collect demos → BC → deploy (fine-tune with RL)
├── Complex contact-rich dynamics
│   └── Simulation-based RL: Domain randomize → sim-to-real
└── Open-ended / language conditioned
    └── VLA: OpenVLA or Octo → fine-tune on your robot
```

### Key Pitfalls

| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| **Covariate shift in BC** | Robot drifts off trajectory | Use DAgger (online corrections) or add noise during training |
| **Reality gap** | Policy fails on real robot | Domain randomization + SysID + real-world fine-tuning |
| **Reward hacking in RL** | High reward in sim, nonsense | Regularize rewards, add safety constraints |
| **Simulation overhead** | Training too slow | Use GPU-accelerated simulators (Isaac Gym, Genesis) |
| **VLA latency** | >500 ms action delay | Quantize model (FP16→INT8), use smaller backbone |
| **Torque-sensor drift** | Poor impedance performance | Auto-calibrate before each task batch |

---

## 11. Cross-References

| Reference | Description |
|-----------|-------------|
| [06-Advanced/01-Multimodal-AI.md] | Vision models for robotics (SAM, DINOv2, CLIP) |
| [01-Foundations/06-Reinforcement-Learning.md] | RL for robot control (PPO, SAC, offline) |
| [06-Advanced/07-Time-Series-Forecasting.md] | Control theory background (PID, LQR, MPC) |
| [08-Reference/01-Glossary.md] | Robotics terms (DoF, EE, PMP, etc.) |
| [01-Foundations/01-Deep-Learning.md] | CNN/Transformer architectures used in perception |
| [06-Advanced/03-Computer-Vision.md] | Depth estimation, segmentation, object detection |
| [01-Foundations/03-Probability-and-Stats.md] | Probabilistic robotics, Kalman filters, particle filters |
| [04-Engineering/*.md] | MLOps, deployment, model serving on edge hardware |

---
*Document version: 2.0 — June 2026 | Tier 2-3: Expanded with code examples, VLA models, and practical guidance*
