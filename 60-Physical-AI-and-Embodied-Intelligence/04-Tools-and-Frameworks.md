# Physical AI and Embodied Intelligence: Tools and Frameworks

> **Category:** 60 — Physical AI and Embodied Intelligence  
> **Last Updated:** July 2026  
> **Cross-references:** [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md), [05-Future-Outlook.md](./05-Future-Outlook.md)

---

## Table of Contents

1. [Robot Operating System (ROS 2)](#1-robot-operating-system-ros-2)
2. [Simulation Platforms](#2-simulation-platforms)
3. [Robot Learning Frameworks](#3-robot-learning-frameworks)
4. [Hardware Platforms](#4-hardware-platforms)
5. [Cloud Robotics Platforms](#5-cloud-robotics-platforms)
6. [AI Model Libraries](#6-ai-model-libraries)
7. [Teleoperation and Data Collection](#7-teleoperation-and-data-collection)
8. [Development Tools and IDEs](#8-development-tools-and-ides)

---

## 1. Robot Operating System (ROS 2)

### 1.1 ROS 2 Overview

ROS 2 is the de facto standard middleware for robotics development:

```
ROS 2 ARCHITECTURE:
─────────────────────────────────────────────────────────────
Application Layer    │ Robot applications, behaviors, tasks
─────────────────────┤
ROS 2 Middleware      │ Topics, Services, Actions, Parameters
─────────────────────┤
DDS Implementation   │ FastDDS, CycloneDDS, Connext
─────────────────────┤
Transport Layer      │ UDP, TCP, Shared Memory
─────────────────────┤
OS Layer             │ Linux, Windows, macOS
─────────────────────────────────────────────────────────────
```

**Key Concepts:**
- **Nodes**: Modular components (perception, planning, control)
- **Topics**: Pub/sub messaging (sensor data, robot state)
- **Services**: Request/reply (parameter queries, configuration)
- **Actions**: Long-running goals (navigation, manipulation)
- **Parameters**: Runtime configuration

### 1.2 Essential ROS 2 Packages for Physical AI

| Package | Purpose | Use Case |
|---------|---------|----------|
| `nav2` | Navigation stack | AMR navigation, SLAM |
| `moveit2` | Motion planning | Manipulation, grasp execution |
| `ros2_control` | Hardware abstraction | Motor control, sensor drivers |
| `gazebo_ros2_control` | Sim integration | Sim-to-real pipeline |
| `vision_msgs` | Perception messages | Object detection, pose estimation |
| `tf2` | Coordinate transforms | Multi-frame robot state |
| `robot_state_publisher` | URDF visualization | Robot model publishing |
| `joint_state_publisher` | Joint state publishing | Robot state estimation |

### 1.3 MoveIt 2: Motion Planning

MoveIt 2 is the standard framework for robot manipulation:

```python
# MoveIt 2 Python example: Pick and place
import rclpy
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped

# Initialize MoveIt
rclpy.init()
moveit = MoveItPy(node_name="pick_and_place")

# Get planning group
arm = moveit.get_planning_group("panda_arm")
hand = moveit.get_planning_group("hand")

# Create pick pose
pick_pose = PoseStamped()
pick_pose.header.frame_id = "panda_link0"
pick_pose.pose.position.x = 0.4
pick_pose.pose.position.y = 0.1
pick_pose.pose.position.z = 0.3
pick_pose.pose.orientation.w = 1.0

# Plan and execute pick
arm.set_pose_target(pick_pose)
plan = arm.plan()
if plan:
    arm.execute(plan)
    hand.execute("open")  # Grasp
    
# Create place pose
place_pose = PoseStamped()
place_pose.pose.position.x = 0.0
place_pose.pose.position.y = -0.3
place_pose.pose.position.z = 0.3
place_pose.pose.orientation.w = 1.0

# Plan and execute place
arm.set_pose_target(place_pose)
plan = arm.plan()
if plan:
    arm.execute(plan)
    hand.execute("close")  # Release
```

### 1.4 Nav2: Autonomous Navigation

```python
# Nav2 Python example: Navigate to goal
from nav2_simple_commander import NavigationEn

navigator = NavigationEn()
navigator.lifecycle_startup()

# Set initial pose
navigator.setInitialPose([0.0, 0.0, 0.0])

# Navigate to goal
navigator.goToPose([1.0, 2.0, 0.0])  # x, y, theta

# Follow waypoints
waypoints = [[1.0, 2.0, 0.0], [2.0, 3.0, 1.57], [3.0, 2.0, 3.14]]
navigator.followWaypoints(waypoints)
```

---

## 2. Simulation Platforms

### 2.1 NVIDIA Isaac Sim / Omniverse

The most comprehensive robot simulation platform:

**Key Features:**
- Photorealistic rendering (RTX ray tracing)
- PhysX 5 physics engine
- Automatic domain randomization
- ROS 2 integration
- Digital twin creation tools

**System Requirements:**
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 2070 | RTX 4090 |
| RAM | 32 GB | 64 GB |
| CPU | 8 cores | 16 cores |
| Storage | 50 GB SSD | 200 GB NVMe |

**Installation:**
```bash
# Install Isaac Sim
pip install isaacsim

# Or use Omniverse Launcher
# Download from https://developer.nvidia.com/isaac-sim

# Launch Isaac Sim
isaacsim.headless  # For batch simulation
isaacsim --no-window  # GUI mode
```

**Python API Example:**
```python
from omni.isaac.kit import SimulationApp

# Initialize simulation
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path

# Create world
world = World(stage_units_in_meters=1.0)

# Load robot
assets_root = get_assets_root_path()
robot = Robot(
    prim_path="/World/robot",
    usd_path=f"{assets_root}/Robots/FrankaEmika/panda_instanceable.usd"
)

# Add environment objects
cuboid = world.scene.add_cuboid(
    name="target_object",
    position=[0.5, 0.0, 0.1],
    size=[0.05, 0.05, 0.05],
    color=[1.0, 0.0, 0.0]
)

# Run simulation
world.reset()
for i in range(1000):
    world.step(render=True)
```

### 2.2 MuJoCo

The fast physics engine preferred for research:

```python
import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Get observations
        joint_pos = data.qpos
        joint_vel = data.qvel
        
        # Apply control
        data.ctrl[:] = desired_torques
        
        # Update viewer
        viewer.sync()
```

**MuJoCo vs. Other Simulators:**

| Feature | MuJoCo | Isaac Sim | PyBullet | Gazebo |
|---------|--------|-----------|----------|--------|
| Speed | 1000x real-time | 100x real-time | 200x real-time | 10-50x real-time |
| Rendering | Basic | Photorealistic | Basic | Medium |
| Physics accuracy | High | High | Medium | Medium |
| Contact modeling | Excellent | Excellent | Good | Good |
| Ease of use | Medium | Hard | Easy | Medium |
| Free | Yes (2022+) | Yes (limited) | Yes | Yes |

### 2.3 Genesis: Ultra-Fast Physics

Newest simulator, designed for AI training:

**Key Innovation:** 10-80x faster than MuJoCo for parallel simulation

```python
import genesis as gs

# Initialize Genesis
gs.init(backend=gs.gpu)

# Create scene with 1000 robots
scene = gs.Scene(
    sim_options=gs.SimOptions(dt=1/100),
    viewer_options=gs.ViewerOptions()
)

# Add 1000 robot instances in parallel
for i in range(1000):
    robot = scene.add_entity(
        gs.morphs.MJCF(file="robot.xml"),
        pos=[i * 0.5, 0, 0]
    )

# Build and run
scene.build()
for step in range(10000):
    scene.step()
    
    # Get all observations in parallel
    observations = [robot.get_qpos() for robot in scene.entities]
```

### 2.4 Other Simulation Platforms

| Platform | Best For | Language | Open Source |
|----------|---------|----------|-------------|
| PyBullet | Quick prototyping | Python | Yes |
| Gazebo (Harmonic) | ROS 2 integration | C++/Python | Yes |
| SAPIEN | Articulated objects | Python | Yes |
| Habitat (Meta) | Navigation, embodied AI | Python/C++ | Yes |
| AI2-THOR (Allen AI) | Indoor simulation | Python | Yes |
| RoboSuite | Manipulation benchmark | Python | Yes |
| robomimic | Imitation learning | Python | Yes |

---

## 3. Robot Learning Frameworks

### 3.1 RoboSuite + robomimic

Standard benchmark for manipulation learning:

```python
import robosuite as suite
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory

# Create environment
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
)

# Load trained policy
config = {...}  # Algorithm config
algo = algo_factory(config)
algo.load_checkpoint("path/to/checkpoint")

# Evaluate policy
obs = env.reset()
for step in range(200):
    action = algo.get_action(obs)
    obs, reward, done, info = env.step(action)
    if done:
        print(f"Task completed in {step} steps!")
        break
```

### 3.2 TorchRL (Meta)

Deep RL library for robotics:

```python
import torch
from torchrl.collectors import MultiSyncDataCollector
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import PPO

# Define environment
from torchrl.envs import GymWrapper
env = GymWrapper(gym.make("PandaPickAndPlace-v1"))

# Define policy
policy = ProbabilisticActor(
    module=nn.Sequential(
        nn.Linear(env.observation_spec.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_spec.shape[0]),
    ),
    dist_class=TanhNormal,
    dist_kwargs={"low": -1, "high": 1}
)

# Train with PPO
collector = MultiSyncDataCollector([env], policy, frames_per_batch=2048)
loss_module = PPO(actor=policy, collector=collector)

for batch in collector:
    loss_vals = loss_module(batch)
    loss_vals["loss_objective"].backward()
    optimizer.step()
```

### 3.3 Tianshou (Tsinghua)

High-performance RL library:

```python
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer

# Define policy
actor = ActorNet(...)
critic = CriticNet(...)
optim = torch.optim.Adam(list(actor) + list(critic), lr=3e-4)

policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=torch.distributions.Normal,
    action_space=env.action_space
)

# Create collectors
train_collector = Collector(policy, train_env, VectorReplayBuffer(20000))
test_collector = Collector(policy, test_env)

# Train
result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=100,
    batch_size=256,
    step_per_epoch=10000,
    step_per_collect=2048,
).run()

print(f"Best reward: {result.best_reward}")
```

### 3.4 Stable Baselines3

Easy-to-use RL library (limited for robotics):

```python
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env("FetchReach-v1", n_envs=4)

# Create and train PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_fetch_reach")
```

### 3.5 Comparison of RL Frameworks

| Framework | Speed | Ease of Use | Robotics Support | Best For |
|-----------|-------|-------------|-----------------|----------|
| TorchRL | Fast | Medium | Excellent | Research, production |
| Tianshou | Fast | Easy | Good | Quick experiments |
| Stable Baselines3 | Medium | Very Easy | Limited | Prototyping |
| RLlib (Ray) | Fast (distributed) | Medium | Good | Large-scale training |
| CleanRL | Fast | Easy | Good | Single-file experiments |
| robomimic | Medium | Easy | Excellent | Imitation learning |

---

## 4. Hardware Platforms

### 4.1 Robot Arms

| Robot | DOF | Payload | Reach | Price | Best For |
|-------|-----|---------|-------|-------|----------|
| Franka Emika Panda | 7 | 3 kg | 855mm | $30K | Research, cobot |
| Universal Robots UR5e | 6 | 5 kg | 850mm | $35K | Industrial |
| Universal Robots UR10e | 6 | 12.5 mm | 1300mm | $45K | Industrial |
| KUKA LBR iiwa 14 | 7 | 14 kg | 820mm | $50K+ | Industrial |
| Robotiq 2F-85 | - | 5 kg | - | $5K | Gripper |
| Robotiq 2F-140 | - | 10 kg | - | $7K | Gripper |
| WidowX 250 | 6 | 250g | 500mm | $500 | Education, light tasks |
| xArm 7 | 7 | 3.5 kg | 700mm | $8K | Prosumer |

### 4.2 Humanoid Robots

| Robot | Height | Weight | DOF | Price | Status |
|-------|--------|--------|-----|-------|--------|
| Figure 02 | 1.7m | 60kg | 42 | $50-100K (est.) | Commercial |
| Tesla Optimus | 1.7m | 57kg | 28+ | $20K (target) | Internal |
| Unitree H1 | 1.8m | 47kg | 19 | $90K | Shipping |
| Unitree G1 | 1.3m | 35kg | 23 | $16K | Shipping |
| Agility Digit | 1.75m | 65kg | 28 | $250K (est.) | Commercial |
| Sanctuary AI Phoenix | 1.7m | 70kg | 27+ | N/A | Prototype |
| 1X NEO | 1.65m | 30kg | 31 | N/A | Prototype |
| Apptronik Apollo | 1.7m | 73kg | 32 | N/A | Prototype |

### 4.3 Mobile Robots

| Robot | Type | Payload | Speed | Price | Use Case |
|-------|------|---------|-------|-------|----------|
| Boston Dynamics Spot | Quadruped | 14 kg | 1.6 m/s | $74.5K | Inspection |
| Unitree B2 | Quadruped | 40 kg | 6 m/s | $30K | Industrial |
| ANYmal X | Quadruped | 15 kg | 1.0 m/s | $150K | Inspection |
| Clearpath Husky | Wheeled | 75 kg | 1.0 m/s | $25K | Research |
| TurtleBot 4 | Wheeled | 1 kg | 0.22 m/s | $3K | Education |
| AgileX Scout | Wheeled | 100 kg | 1.5 m/s | $8K | Research |
| Amazon Proteus | AMR | 100+ kg | 1.5 m/s | N/A | Warehouse |

### 4.4 Grippers and End-Effectors

| Gripper | Type | Force | Stroke | Price |
|---------|------|-------|--------|-------|
| Robotiq 2F-85 | Adaptive | 70N | 85mm | $5K |
| Robotiq 2F-140 | Adaptive | 220N | 140mm | $7K |
| OnRobot RG2 | Adaptive | 40N | 115mm | $3K |
| OnRobot VG10 | Vacuum | 10×20N | N/A | $4K |
| Robotiq 3-Finger | Dexterous | 60N | N/A | $12K |
| Schunk EGP | Parallel | 145N | 100mm | $8K |
| Barrett Hand | Dexterous | 10N/finger | N/A | $20K+ |
| LEAP Hand | Dexterous | 5N/finger | 80mm | $500 |

### 4.5 Sensors

| Sensor | Type | Range | Price | Use Case |
|--------|------|-------|-------|----------|
| Intel RealSense D435i | RGB-D | 0.2-10m | $300 | Manipulation |
| Intel RealSense L515 | LiDAR RGB-D | 0.25-9m | $500 | Indoor navigation |
| Livox Mid-360 | LiDAR | 40m | $600 | Outdoor navigation |
| Ouster OS1-32 | LiDAR | 120m | $8K | Autonomous vehicles |
| Intel RealSense T265 | Tracking | N/A | $200 | Visual odometry |
| Force/Torque ATI | F/T sensor | ±2400N | $5K | Force control |
| GelSight Mini | Tactile | N/A | $500 | Dexterous manipulation |
| BioTac | Tactile | N/A | $5K | Research |

---

## 5. Cloud Robotics Platforms

### 5.1 AWS RoboMaker

```python
import boto3

# Create robot application
robomaker = boto3.client('robomaker')

# Create robot application
response = robomaker.create_robot_application(
    name='my-manipulation-robot',
    robotSoftwareSuite={
        'name': 'ROS2',
        'version': 'Humble'
    },
    sources=[
        {
            'architecture': 'X86_64',
            's3Bucket': 'my-robot-bucket',
            's3Key': 'robot-app.zip'
        }
    ]
)

# Create simulation application
response = robomaker.create_simulation_application(
    name='my-sim-app',
    robotSoftwareSuite={
        'name': 'ROS2',
        'version': 'Humble'
    },
    simulationSoftwareSuite={
        'name': 'Gazebo',
        'version': 'Harmonic'
    },
    sources=[
        {
            'architecture': 'X86_64',
            's3Bucket': 'my-robot-bucket',
            's3Key': 'sim-app.zip'
        }
    ]
)

# Run simulation job
response = robomaker.create_simulation_job(
    maxJobDurationInSeconds=3600,
    iamRole='arn:aws:iam::role/RoboMakerRole',
    robotApplications=[
        {
            'application': 'arn:aws:robomaker:us-east-1:123456789:robot-app/my-app',
            'applicationVersion': '\$LATEST'
        }
    ],
    simulationApplications=[
        {
            'application': 'arn:aws:robomaker:us-east-1:123456789:sim-app/my-sim',
            'applicationVersion': '\$LATEST'
        }
    ]
)
```

### 5.2 NVIDIA Isaac Lab

Cloud-based robot training:

**Features:**
- Distributed GPU simulation
- Automatic domain randomization
- Sim-to-real validation
- Fleet management

### 5.3 Google Cloud Robotics

```python
from google.cloud import robotics_v1

client = robotics_v1.RobotControllerServiceClient()

# List robot clusters
parent = "projects/my-project/locations/us-central1"
clusters = client.list_robot_clusters(parent=parent)

for cluster in clusters:
    print(f"Cluster: {cluster.name}")
    print(f"  Machines: {cluster.fleet.machines_count}")
```

---

## 6. AI Model Libraries

### 6.1 Hugging Face for Robotics

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load VLA model
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b")

# Inference
inputs = processor(images=robot_camera_image, text="Pick up the red cup")
outputs = model.generate(**inputs, max_new_tokens=100)
```

### 6.2 NVIDIA Isaac GR00T

```python
from isaac_gr00t import GR00TModel

# Load GR00T model
model = GR00TModel.from_pretrained("nvidia/gr00t-n1")

# Predict actions from observations
observations = {
    "image": camera_image,
    "proprioception": joint_positions,
    "instruction": "Pick up the red cup"
}

actions = model.predict(observations)
robot.execute(actions)
```

### 6.3 Key Model Hubs

| Platform | Models | Robotics Focus | License |
|----------|--------|---------------|---------|
| Hugging Face | 500K+ | Growing | Mixed |
| NVIDIA NGC | 100+ | Excellent | Various |
| TorchHub | 100+ | Limited | BSD |
| PyTorch Hub | 50+ | Limited | BSD |
| GitHub | 1000+ | Varies | Varies |

---

## 7. Teleoperation and Data Collection

### 7.1 Teleoperation Interfaces

| Interface | Bandwidth | Immersion | Cost | Use Case |
|-----------|----------|-----------|------|----------|
| VR headset + controllers | High | Very High | $500-1000 | Dexterous manipulation |
| Keyboard/mouse | Low | Low | $0 | Simple tasks |
| SpaceMouse | Medium | Low | $200 | Arm positioning |
| Master-slave (ALOHA) | High | High | $5-20K | Bimanual manipulation |
| Exoskeleton | Very High | Very High | $10-50K | Full-body control |

### 7.2 ALOHA: Low-Cost Teleoperation

```python
# ALOHA-style teleoperation
import time
import numpy as np

class ALOHATeleoperation:
    def __init__(self, leader_arms, follower_arms):
        self.leader = leader_arms  # 2x 6-DOF arms with grippers
        self.follower = follower_arms  # 2x 7-DOF arms
        
    def collect_demonstration(self, num_steps=1000):
        """Record teleoperation demonstration"""
        observations = []
        actions = []
        
        for step in range(num_steps):
            # Read leader arm positions
            leader_pos = np.concatenate([
                self.leader[0].get_joint_positions(),
                self.leader[1].get_joint_positions()
            ])
            
            # Command follower arms
            self.follower[0].set_joint_positions(leader_pos[:7])
            self.follower[1].set_joint_positions(leader_pos[7:14])
            
            # Record data
            obs = {
                "images": {
                    "top": self.get_camera("top"),
                    "wrist_left": self.get_camera("wrist_left"),
                    "wrist_right": self.get_camera("wrist_right")
                },
                "qpos": self.get_follower_positions()
            }
            action = leader_pos
            
            observations.append(obs)
            actions.append(action)
            
            time.sleep(0.02)  # 50 Hz
        
        return observations, actions
```

### 7.3 Data Collection Best Practices

| Practice | Description |
|----------|-------------|
| Record at 30+ Hz | High frequency for smooth trajectories |
| Multiple camera views | At least 2-3 angles |
| Record language annotations | Natural language task descriptions |
| Varied demonstrations | Different approaches to same task |
| Failure demonstrations | Show what NOT to do |
| Perturbation demos | Show recovery from errors |
| Consistent lighting | Reduce visual variability |
| Clean backgrounds | Reduce visual distraction |

---

## 8. Development Tools and IDEs

### 8.1 IDE and Editor Support

| Tool | ROS 2 Support | Debugging | Simulation | Price |
|------|--------------|-----------|-----------|-------|
| VS Code + ROS extension | Excellent | Good | Limited | Free |
| CLion | Good | Excellent | Limited | $150/yr |
| PyCharm | Good | Good | Limited | Free/Pro |
| Visual Studio | Good | Excellent | Limited | Free/Pro |

### 8.2 ROS 2 Development Tools

```bash
# Build ROS 2 workspace
colcon build --symlink-install

# Run tests
colcon test
colcon test-result --all

# Launch robot
ros2 launch my_robot_bringup robot.launch.py

# Visualize in RViz2
rviz2

# Inspect topics
ros2 topic list
ros2 topic echo /robot/joint_states

# Record bag files
ros2 bag record -a -o robot_data
ros2 bag play robot_data
```

### 8.3 Debugging and Profiling

```python
# ROS 2 debugging with ros2_control
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 control reload_controller_config controller_name

# Performance profiling
ros2 topic hz /robot/camera/image_raw
ros2 topic delay /robot/joint_states

# Log analysis
ros2 run rqt_console rqt_console
ros2 run rqt_graph rqt_graph
```

### 8.4 CI/CD for Robotics

```yaml
# GitHub Actions for ROS 2 robot
name: Robot CI/CD
on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    container: ros:humble
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install dependencies
      run: |
        apt-get update
        rosdep install --from-paths src --ignore-src -r -y
    
    - name: Build
      run: |
        colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
    
    - name: Test
      run: |
        colcon test
        colcon test-result --all
    
    - name: Simulate
      run: |
        # Run Gazebo simulation test
        timeout 120 ros2 launch my_robot sim_test.launch.py
```

---

## Summary: Recommended Stack for Physical AI Development

```
RECOMMENDED DEVELOPMENT STACK (2026):
─────────────────────────────────────────────────────────────
Layer              │ Recommended Tools
─────────────────────────────────────────────────────────────
Robot OS           │ ROS 2 Humble/Iron
Simulation         │ NVIDIA Isaac Sim + MuJoCo
Robot Learning     │ robomimic + TorchRL
AI Models          │ Hugging Face + NVIDIA GR00T
Edge Computing     │ NVIDIA Jetson Orin
Teleoperation      │ ALOHA / VR controllers
Cloud              │ AWS RoboMaker
Version Control    │ Git + LFS (for datasets)
IDE                │ VS Code + ROS extension
─────────────────────────────────────────────────────────────
```

---

*This document is part of the AiBaseKnowledge library. See [01-Overview.md](./01-Overview.md) for the full overview.*
