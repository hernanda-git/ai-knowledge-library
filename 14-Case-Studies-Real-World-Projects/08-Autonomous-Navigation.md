# 08 — Autonomous Navigation (Warehouse Robot)

## Case Study: SLAM + Path Planning + RL for Warehouse Automation

| Metadata | Value |
|----------|-------|
| **Industry** | Logistics / Robotics / Warehouse Automation |
| **Domain** | Autonomous navigation, SLAM, path planning |
| **Difficulty** | Expert |
| **Est. Timeline** | 16-32 weeks |
| **Team Size** | 8-12 engineers (3 robotics, 2 ML, 2 simulation, 2 hardware, 1 safety) |

---

## 🎯 Problem Statement

### Business Context

**Company:** LogiBot Robotics (100,000 sq.ft warehouse, 50K SKUs, 15M picks/month)
**Current Fleet:** 30 human-driven forklifts + 12 AGVs (magnetic tape guided)
**Shifts:** 3 shifts (20 hrs/day, 4 hrs for charging/maintenance)

### Pain Points

1. **Fixed Path Inefficiency** — Magnetic tape AGVs follow fixed routes; traffic congestion causes 25% time waste
2. **Poor Adaptability** — Re-laying tape for layout changes costs $12K/week; takes 2 days
3. **Human-Robot Co-existence** — 12 safety incidents/year where AGVs didn't properly detect humans
4. **Scalability** — Adding new AGV requires tape-laying + reprogramming; takes 1 week
5. **Pick Density** — Robots must navigate narrow aisles (1.2m wide) with 2m pallet clearance
6. **Charging Management** — Reactive charging leads to 15% fleet downtime during peak hours
7. **Sim-to-Real Gap** — AGVs trained in simulation fail in real warehouse due to lighting, floor variance

### Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| **Navigation Success Rate** | > 99% | 95% |
| **Path Efficiency** | +30% shorter routes | Fixed tape paths |
| **Human Detection Accuracy** | 99.9% | 95% |
| **Throughput (picks/hour)** | +40% | Baseline AGV |
| **Deployment Time (new robot)** | < 4 hours | 1 week |
| **Fleet Uptime** | > 98% | 85% |
| **Safety Incidents/Year** | < 2 | 12 |

---

## 🏗️ Solution Architecture

### High-Level System

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          FLEET MANAGEMENT CLOUD                                     │
│                                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │  Fleet Orchestrator  │  │  Warehouse Mgmt Sys  │  │  Mission Planner           │  │
│  │  (task assignment,   │  │  (inventory, orders)  │  │  (multi-robot routing,    │  │
│  │   traffic manager)   │  └─────────────────────┘  │   deadlock prevention)     │  │
│  └──────────┬──────────┘                            └─────────────┬───────────────┘  │
│             │                                                     │                  │
└─────────────┼─────────────────────────────────────────────────────┼──────────────────┘
              │                          WiFi (ROS2 over DDS)        │
┌─────────────┼─────────────────────────────────────────────────────┼──────────────────┐
│             ▼                                                     ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         ONBOARD ROBOT STACK (each robot)                      │   │
│  │                                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    ROS2 HUMBLE (Middleware Layer)                        │  │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                               │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │   │
│  │  │  PERCEPTION      │  │  LOCALIZATION    │  │  PLANNING        │            │   │
│  │  │                  │  │                  │  │                  │            │   │
│  │  │  2× LiDAR (Sick) │  │  SLAM Toolbox    │  │  Global Planner  │            │   │
│  │  │  4× RGB-D Cam    │  │  (Cartographer)  │  │  (A* Hybrid)    │            │   │
│  │  │  12× Ultrasonic  │  │                  │  │                  │            │   │
│  │  │  IMU + Wheel     │  │  AMCL (adaptive) │  │  Local Planner   │            │   │
│  │  │  Odometry        │  │                  │  │  (TEB + RL)     │            │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘            │   │
│  │           │                     │                      │                     │   │
│  │           ▼                     ▼                      ▼                     │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  CONTROL STACK                                                      │    │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │    │   │
│  │  │  │  Object     │  │  Collision  │  │  Velocity   │  │  Motor    │  │    │   │
│  │  │  │  Detection  │  │  Avoidance  │  │  Commands   │  │  Control  │  │    │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │    │   │
│  │  └──────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  HARDWARE ABSTRACTION LAYER                                          │    │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │   │
│  │  │  │  Motors  │  │  Battery │  │  Safety  │  │  LEDs/   │            │    │   │
│  │  │  │  (4×)   │  │  BMS     │  │  PLC     │  │  Buzzer  │            │    │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │    │   │
│  │  └──────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### SLAM Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LiDAR       │     │  Scan        │     │  Scan        │     │  Pose        │
│  (40Hz)      │────▶│  Matching    │────▶│  Registration │────▶│  Estimation  │
│              │     │  (ICP)       │     │  (graph SLAM) │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  RGB-D Cam   │     │  Feature     │     │  Visual      │            │
│  (30Hz)      │────▶│  Extraction  │────▶│  Odometry    │────────────┘
│              │     │  (ORB/SIFT)  │     │  (ORB-SLAM3) │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
┌──────────────┐     ┌──────────────┐              │
│  IMU         │     │  Pre-        │              │
│  (200Hz)     │────▶│  integration │──────────────┘
│  + Wheel Enc │     │  (ESKF)      │
└──────────────┘     └──────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │  MAP (occupancy grid + feature map) │
                              │  - Updated continuously              │
                              │  - Shared across fleet via cloud     │
                              └─────────────────────────────────────┘
```

### Deep RL Path Planning

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RL-BASED LOCAL PLANNER                            │
│                                                                      │
│  Observation Space:                                                  │
│  - LiDAR scan (360°, 360 values)                                    │
│  - Relative goal position (dx, dy, theta)                           │
│  - Current velocity (v_linear, v_angular)                           │
│  - Occupancy grid crop (32×32 around robot)                         │
│                                                                      │
│  Action Space:                                                       │
│  - Linear velocity [-0.8, 1.5] m/s                                  │
│  - Angular velocity [-1.0, 1.0] rad/s                               │
│                                                                      │
│  Reward Function:                                                    │
│  - Reaching goal:              +100                                  │
│  - Collision:                   -50                                  │
│  - Step penalty:                -0.1 × time_step                     │
│  - Progress toward goal:       +5 × Δ_distance_to_goal              │
│  - Jerk penalty:               -0.5 × Δ_acceleration                │
│  - Human proximity (< 1m):      -20                                  │
│  - Close to obstacles:         -2 × (1 / min_obstacle_distance)     │
│                                                                      │
│  Algorithm: PPO (Proximal Policy Optimization)                       │
│  - Actor-critic network: 3-layer MLP (256, 128, 64)                 │
│  - LiDAR input: CNN (3 conv layers) → 128-dim embedding            │
│  - Learning rate: 3e-4                                              │
│  - Clip epsilon: 0.2                                                │
│  - Entropy coefficient: 0.01                                        │
│  - Training steps: 10M (≈ 2 days on 8× RTX 4090)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Robot Middleware** | ROS2 Humble | Humble | Industry standard, real-time, DDS |
| **SLAM** | Cartographer + SLAM Toolbox | 2.0 / 2.0 | 2D LiDAR SLAM, loop closure |
| **Global Planner** | Nav2 + A* Hybrid | 1.2 | Smooth A* on costmap |
| **Local Planner** | TEB (Timed Elastic Band) + RL | Custom | Smooth, constraint-aware |
| **RL Framework** | Stable-Baselines3 + SB3-Contrib | 2.3 / 2.1 | PPO implementation |
| **Simulation** | Gazebo Classic + Ignition Gazebo | 11 / 7 | Physics simulation |
| **Sim-to-Real** | Domain Randomization + NVIDIA Isaac Sim | — | Bridge gap |
| **Perception** | PointPillars (LiDAR) + YOLOv8 (RGB-D) | 3D / 8.2 | Object detection |
| **Fleet Orchestration** | OpenRMF + custom fleet manager | 2.0 | Multi-robot coordination |
| **Edge/Onboard** | NVIDIA Jetson Orin AGX | — | 275 TOPS, low power |
| **Safety PLC** | Sick PLB (Safety Laser Scanner) | — | SIL2/PLd certified |
| **Monitoring** | ROS2 Monitoring + Grafana | — | Fleet telemetry |
| **Infrastructure** | Docker + Ansible + Kubernetes | — | Fleet software updates |

### Installation

```bash
# ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop ros-humble-navigation2
sudo apt install ros-humble-slam-toolbox ros-humble-cartographer
sudo apt install ros-humble-teb-local-planner

# RL & Simulation
pip install stable-baselines3[extra]==2.3.2 sb3-contrib==2.1.0
pip install gymnasium[box2d]==0.29.1
sudo apt install ros-humble-gazebo-ros-pkgs

# Perception
pip install torch==2.1.2 ultralytics==8.2.0
pip install open3d==0.18.0

# Fleet management
pip install rmf-fleet-adapter-python==2.0.0
```

---

## ⚙️ Implementation Details

### 1. SLAM with Cartographer

```python
# src/localization/slam_node.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros
import numpy as np

class SLAMNode(Node):
    """ROS2 node wrapping Cartographer SLAM for warehouse mapping."""

    def __init__(self):
        super().__init__("slam_node")
        self.map_publisher = self.create_publisher(
            OccupancyGrid, "/map", 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Cartographer bridge
        from cartographer_ros.msg import SubmapList
        self.submap_list_sub = self.create_subscription(
            SubmapList, "/submap_list", self.submap_callback, 10
        )

        self.map_data = None
        self.robot_pose = None
        self.get_logger().info("SLAM Node initialized")

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan for SLAM update."""
        # Cartographer handles this internally via ROS2 plugin
        # This node mainly monitors and publishes the map
        pass

    def odom_callback(self, msg: Odometry):
        """Update odometry estimate."""
        self.robot_pose = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "theta": self._quaternion_to_yaw(
                msg.pose.pose.orientation
            ),
        }

    def submap_callback(self, msg):
        """Receive updated map from Cartographer."""
        # Cartographer publishes submap updates automatically
        pass

    def get_current_map(self) -> OccupancyGrid:
        """Retrieve current occupancy grid map."""
        # In production: read from /map topic
        # For template: return stored map or wait
        return self.map_data

    @staticmethod
    def _quaternion_to_yaw(quat) -> float:
        """Convert ROS2 quaternion to yaw angle."""
        import math
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
```

### 2. A* Global Planner with Warehouse Constraints

```python
# src/planning/global_planner.py
import heapq
import numpy as np
from typing import List, Tuple, Optional

class WarehouseAStar:
    """A* path planner with warehouse-specific constraints.

    Handles: narrow aisles, pallet obstacles, one-way zones, charging stations.
    """

    def __init__(self, costmap: np.ndarray, resolution: float = 0.05):
        """
        Args:
            costmap: 2D occupancy grid (0=free, 100=occupied, 255=unknown)
            resolution: meters per cell
        """
        self.costmap = costmap
        self.resolution = resolution
        self.height, self.width = costmap.shape

        # Warehouse zones
        self.one_way_zones = []  # (x_min, x_max, y_min, y_max, direction)
        self.charging_stations = []

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_iterations: int = 100000,
    ) -> Optional[List[Tuple[float, float]]]:
        """Compute optimal path respecting warehouse constraints."""

        start_cell = self._world_to_grid(start)
        goal_cell = self._world_to_grid(goal)

        if not self._is_valid(start_cell) or not self._is_valid(goal_cell):
            self.get_logger().warn("Start or goal in occupied cell")
            return None

        # A* with weighted heuristic
        open_set = []
        heapq.heappush(open_set, (0.0, start_cell))
        came_from = {}
        g_score = {start_cell: 0.0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while open_set and len(came_from) < max_iterations:
            _, current = heapq.heappop(open_set)

            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

    def _cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """Compute movement cost with warehouse constraints."""
        # Base Euclidean distance
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Costmap cost (avoid obstacles)
        cell_cost = self.costmap[neighbor[1], neighbor[0]]
        obstacle_cost = 0.0
        if cell_cost > 50:  # Near obstacle
            obstacle_cost = 10.0 * (cell_cost / 100.0)
        if cell_cost >= 100:  # Occupied
            return float("inf")

        # One-way zone penalty (wrong direction)
        for zone in self.one_way_zones:
            if self._in_zone(neighbor, zone):
                # Check if movement direction matches allowed direction
                allowed_dir = zone[4]  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
                if not self._direction_allowed(dx, dy, allowed_dir):
                    obstacle_cost += 50.0

        # Charging station bonus (slight preference if battery low)
        # (Simplified: positive cost to avoid unnecessary detours)
        for station in self.charging_stations:
            if self._in_station(neighbor, station):
                obstacle_cost -= 5.0

        return dist + obstacle_cost

    def _heuristic(
        self, cell: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        """Weighted Manhattan heuristic for warehouse."""
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        # Diagonal shortcut cost
        return min(dx, dy) * np.sqrt(2) + abs(dx - dy)

    def _get_neighbors(
        self, cell: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """8-connected neighbors (including diagonals)."""
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cell[0] + dx, cell[1] + dy
                if self._is_valid((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors

    def _is_valid(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        return (0 <= x < self.width and 0 <= y < self.height
                and self.costmap[y, x] < 100)

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        return (int(world_pos[0] / self.resolution),
                int(world_pos[1] / self.resolution))

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        return (grid_pos[0] * self.resolution,
                grid_pos[1] * self.resolution)

    def _reconstruct_path(
        self, came_from: dict, current: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        path = []
        while current in came_from:
            path.append(self._grid_to_world(current))
            current = came_from[current]
        path.append(self._grid_to_world(current))
        path.reverse()
        return path
```

### 3. RL Local Planner with PPO

```python
# src/planning/rl_planner.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional

class WarehouseRobotEnv(gym.Env):
    """Custom Gym environment for warehouse robot navigation.

    Observation: LiDAR scan (360) + goal vector (3) + velocity (2)
    Action: linear velocity, angular velocity
    """

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}

        # Observation space: LiDAR (360) + goal (3) + velocity (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(365,), dtype=np.float32
        )

        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-0.8, -1.0]),
            high=np.array([1.5, 1.0]),
            dtype=np.float32
        )

        # Simulation state (in production: Gazebo bridge)
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.goal_pos = np.array([10.0, 5.0])
        self.lidar_scan = np.zeros(360)
        self.velocity = np.array([0.0, 0.0])
        self.step_count = 0
        self.max_steps = 500
        self.collision = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Random start and goal positions
        self.robot_pos = self.np_random.uniform(
            low=[0, 0, -np.pi], high=[20, 20, np.pi]
        )
        self.goal_pos = self.np_random.uniform(
            low=[0, 0], high=[20, 20]
        )
        self.lidar_scan = np.ones(360) * 5.0  # 5m max range
        self.velocity = np.array([0.0, 0.0])
        self.step_count = 0
        self.collision = False
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        linear_vel, angular_vel = action

        # Update robot pose (simplified kinematics)
        dt = 0.1
        self.robot_pos[0] += linear_vel * np.cos(self.robot_pos[2]) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_pos[2]) * dt
        self.robot_pos[2] += angular_vel * dt
        self.robot_pos[2] = np.arctan2(
            np.sin(self.robot_pos[2]), np.cos(self.robot_pos[2])
        )
        self.velocity = np.array([linear_vel, angular_vel])

        # Simulate LiDAR (simplified)
        self.lidar_scan = self._simulate_lidar()

        # Compute reward
        reward, done, info = self._compute_reward()

        return self._get_obs(), reward, done, False, info

    def _simulate_lidar(self) -> np.ndarray:
        """Simplified LiDAR simulation with random obstacles."""
        scan = np.ones(360) * 5.0
        # Add some random obstacles for training variety
        num_obstacles = self.np_random.integers(0, 5)
        for _ in range(num_obstacles):
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(0.5, 4.0)
            idx = int(angle * 180 / np.pi) % 360
            scan[idx] = min(scan[idx], distance)
        return scan

    def _compute_reward(self):
        dist_to_goal = np.linalg.norm(self.robot_pos[:2] - self.goal_pos)
        prev_dist = getattr(self, "_prev_dist", dist_to_goal + 0.1)
        progress = prev_dist - dist_to_goal

        reward = 0.0
        done = False
        info = {}

        # Goal reached
        if dist_to_goal < 0.3:
            reward += 100.0
            info["success"] = True
            done = True
        else:
            reward += progress * 5.0  # Reward progress

        # Collision penalty (simplified: if LiDAR min < 0.3m)
        min_distance = np.min(self.lidar_scan)
        if min_distance < 0.3:
            reward -= 50.0
            self.collision = True
            info["collision"] = True
            done = True

        # Time penalty
        reward -= 0.1

        # Step limit
        if self.step_count >= self.max_steps:
            done = True
            info["timeout"] = True

        self._prev_dist = dist_to_goal
        return reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        goal_vec = self.goal_pos - self.robot_pos[:2]
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.robot_pos[2]
        return np.concatenate([
            self.lidar_scan,
            [goal_dist * np.cos(goal_angle), goal_dist * np.sin(goal_angle)],
            self.velocity,
        ]).astype(np.float32)


class RLNavigationTrainer:
    """Train PPO policy for warehouse robot navigation."""

    def train(self, total_timesteps: int = 10_000_000):
        env = DummyVecEnv([lambda: WarehouseRobotEnv()])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/rl_nav/",
        )

        model.learn(total_timesteps=total_timesteps)
        model.save("./models/rl_local_planner")
        return model

    def evaluate(self, model_path: str, num_episodes: int = 100):
        """Evaluate trained policy."""
        model = PPO.load(model_path)
        env = WarehouseRobotEnv()

        successes = 0
        collisions = 0
        total_reward = 0.0
        path_lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                steps += 1

            total_reward += episode_reward
            path_lengths.append(steps)

            if info.get("success"):
                successes += 1
            if info.get("collision"):
                collisions += 1

        print(f"Evaluation Results ({num_episodes} episodes):")
        print(f"  Success Rate: {successes/num_episodes:.1%}")
        print(f"  Collision Rate: {collisions/num_episodes:.1%}")
        print(f"  Avg Reward: {total_reward/num_episodes:.1f}")
        print(f"  Avg Path Length: {np.mean(path_lengths):.1f} steps")

        return {
            "success_rate": successes / num_episodes,
            "collision_rate": collisions / num_episodes,
            "avg_reward": total_reward / num_episodes,
            "avg_path_length": np.mean(path_lengths),
        }
```

### 4. ROS2 Navigation Node

```python
# src/navigation/navigation_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np

class RobotNavigator(Node):
    """High-level navigation controller for warehouse robot."""

    def __init__(self):
        super().__init__("robot_navigator")
        self.nav_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.current_pose = None
        self.current_goal = None
        self.navigation_active = False

        # Wait for Nav2 action server
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for Nav2 action server...")

        self.get_logger().info("Navigator initialized")

    def navigate_to(self, x: float, y: float, theta: float = 0.0):
        """Send navigation goal to Nav2."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2)
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.current_goal = goal_msg
        self.navigation_active = True

        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2")
            self.navigation_active = False
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f"Navigation completed: {result}"
        )
        self.navigation_active = False

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Distance remaining
        if hasattr(feedback, "distance_remaining"):
            self.get_logger().debug(
                f"Distance remaining: {feedback.distance_remaining:.2f}m"
            )

    def stop(self):
        """Emergency stop."""
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        self.navigation_active = False
        self.get_logger().warn("EMERGENCY STOP")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
```

### 5. Simulation-to-Real Bridge

```python
# src/simulation/domain_randomization.py
class DomainRandomization:
    """Apply randomizations to simulation parameters to improve sim-to-real transfer."""

    RANDOMIZATIONS = {
        "physics": {
            "friction": (0.3, 0.8),       # Floor friction
            "mass": (0.8, 1.5),           # Robot mass multiplier
            "motor_torque": (0.7, 1.3),   # Motor strength
            "wheel_diameter": (0.95, 1.05),  # Wheel wear
        },
        "sensors": {
            "lidar_noise_std": (0.01, 0.05),  # LiDAR measurement noise
            "lidar_dropout": (0.0, 0.05),     # Random scan dropout
            "camera_exposure": (0.5, 1.5),    # Lighting variation
            "imu_bias": (-0.1, 0.1),          # IMU bias
        },
        "environment": {
            "floor_texture": 3,        # Random floor pattern
            "lighting_intensity": (0.5, 1.5),
            "obstacle_positions": True,  # Randomize shelf positions
            "dynamic_obstacles": True,    # Spawn moving humans
        }
    }

    def apply(self, simulation_world):
        """Apply domain randomization to a Gazebo/Isaac world."""
        import random
        applied = {}

        # Physics randomization
        for param, (low, high) in self.RANDOMIZATIONS["physics"].items():
            value = random.uniform(low, high)
            simulation_world.set_physics_property(param, value)
            applied[f"physics.{param}"] = value

        # Sensor noise
        for param, (low, high) in self.RANDOMIZATIONS["sensors"].items():
            value = random.uniform(low, high)
            simulation_world.set_sensor_noise(param, value)
            applied[f"sensor.{param}"] = value

        # Environment randomization
        if self.RANDOMIZATIONS["environment"]["obstacle_positions"]:
            simulation_world.randomize_obstacle_positions(seed=random.randint(0, 1000))

        self.get_logger().info(f"Applied {len(applied)} domain randomizations")
        return applied

    def curriculum_learning(
        self, episode: int, max_episodes: int = 10000
    ) -> dict:
        """Gradually increase randomization difficulty."""
        progress = min(episode / max_episodes, 1.0)

        # Start easy, gradually increase
        randomization_strength = 0.2 + 0.8 * progress  # 20% → 100%

        applied = {}
        for category, params in self.RANDOMIZATIONS.items():
            if isinstance(params, dict):
                for param, (low, high) in params.items():
                    value = low + (high - low) * randomization_strength
                    applied[f"{category}.{param}"] = value

        return applied
```

---

## 📊 Metrics & Results

### Navigation Performance

| Metric | Baseline (Magnetic Tape) | New System (SLAM + RL) | Delta |
|--------|------------------------|----------------------|-------|
| **Navigation Success Rate** | 95.0% | 99.8% | +4.8 pp |
| **Path Length (avg)** | Fixed (tape) | 32% shorter | -32% |
| **Travel Time (avg)** | 4.2 min/pick | 2.1 min/pick | -50% |
| **Collision Rate** | 0.8% | 0.02% | -97.5% |
| **Human Detection Rate** | 95.0% | 99.9% | +4.9 pp |
| **Replan Time (obstacle)** | N/A (follows tape) | 150ms | N/A |
| **Map Update (layout change)** | 2 days | 5 minutes | -99.8% |

### Fleet Performance (30 robots)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **Picks per Hour (fleet)** | 280 | 785 | +180% |
| **Fleet Uptime** | 85% | 97% | +12 pp |
| **Charging Downtime** | 15% | 5% | -10 pp |
| **Daily Throughput** | 6,720 picks | 18,840 picks | +180% |
| **Deadlock Events** | 45/day | 3/day | -93% |
| **Deployment Time (new robot)** | 1 week | 2 hours | -96% |

### ROI Analysis

```
┌──────────────────────────────────────────────────────────────┐
│  Investment                                                    │
│  Hardware Retrofit (30 robots × $45K)          $1,350,000     │
│  Compute (30× Jetson Orin)                       $150,000     │
│  Engineering + Training                         $800,000      │
│  Safety Certification                            $200,000      │
│  Total Investment                               $2,500,000     │
│                                                                │
│  Annual Savings                                                  │
│  Labor Replacement (25 FTE × $52K)               $1,300,000     │
│  Throughput Increase (180%)                      $4,200,000     │
│  Safety Incident Reduction                       $500,000       │
│  Maintenance Reduction                           $300,000       │
│  Layout Change Savings                           $624,000       │
│  Total Annual Savings                            $6,924,000     │
│                                                                │
│  Payback Period                                    4.3 months  │
│  Year-1 ROI                                          ~177%     │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Sim-to-real with domain randomization** — Starting at 20% randomization and scaling to 100% over training was critical. Policy transferred with only 3% performance drop.

2. **Global A* + RL local** — Pure RL didn't work for long-horizon navigation. Combining A* (global) with RL (local obstacle avoidance) was the winning approach.

3. **Fleet-wide map sharing** — When one robot mapped a new area, all robots benefited immediately. SLAM maps shared via cloud reduced mapping time by 80%.

4. **Safety PLC as hard stop** — Software safety layers caught 99% of issues, but having a SIL2-certified hardware safety PLC for the remaining 1% was essential for regulatory approval.

### ❌ What Went Wrong

1. **Initial RL training in pure simulation failed** — 10M steps in perfect simulation → policy that didn't work in real warehouse. Domain randomization fixed this, but cost 2 weeks.

2. **Floor texture caused LiDAR issues** — Polished concrete floor caused LiDAR beam scattering. Required adding floor-reflectance filtering and upward-tilted LiDAR mounts.

3. **Battery management was an afterthought** — Initially didn't plan for charging scheduling. Added predictive charging (based on route energy estimation) in v2.

4. **Multi-robot deadlocks** — Robots occasionally faced each other in narrow aisles with no way to pass. Solved with "right-of-way" rules + bidirectional aisle protocol.

### ⚠️ Critical Warnings

```
! WARNING: Safety must be hardware-certified (SIL2/PLd minimum).
! WARNING: Human detection is a hard requirement — 99.9% is the minimum.
! WARNING: Simulation alone is insufficient — real-world testing is mandatory.
! WARNING: Battery management is critical for fleet uptime — don't ignore it.
! WARNING: Network latency affects coordination — ROS2 over wired is preferred.
```

### Deployment Checklist

- [ ] Safety PLC (SICK PLB) certified and tested
- [ ] All LiDAR/camera mounting positions validated
- [ ] Floor mapping completed (scan 10 passes for loop closure)
- [ ] Emergency stop buttons tested within 2m reach
- [ ] Human detection sensitivity calibrated (no false negatives)
- [ ] Network coverage (wifi 6) tested in all warehouse areas
- [ ] Charging station positions optimized for fleet flow

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-AUTONOMOUS-NAVIGATION/
├── README.md
├── Makefile
├── requirements.txt
├── .env.example
│
├── configs/
│   ├── config.yaml
│   ├── robot_config.yaml
│   ├── mapping_config.yaml
│   ├── planning_config.yaml
│   ├── safety_config.yaml
│   └── fleet_config.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── localization/
│   │   ├── __init__.py
│   │   ├── slam_node.py
│   │   ├── amcl_node.py
│   │   ├── ekf_localization.py
│   │   └── landmark_detector.py
│   │
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── lidar_processor.py
│   │   ├── human_detector.py
│   │   ├── obstacle_tracker.py
│   │   └── depth_estimator.py
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── global_planner.py
│   │   ├── local_planner.py
│   │   ├── rl_planner.py
│   │   └── deadlock_resolver.py
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   ├── velocity_controller.py
│   │   ├── motor_driver.py
│   │   ├── safety_monitor.py
│   │   └── emergency_stop.py
│   │
│   ├── fleet/
│   │   ├── __init__.py
│   │   ├── fleet_manager.py
│   │   ├── traffic_controller.py
│   │   ├── charging_manager.py
│   │   └── mission_scheduler.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── domain_randomization.py
│   │   ├── curriculum_learning.py
│   │   ├── gazebo_bridge.py
│   │   └── sim_evaluator.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── telemetry.py
│   │   ├── fleet_dashboard.py
│   │   └── alerts.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── geometry.py
│       └── types.py
│
├── tests/
│   ├── unit/
│   │   ├── test_astar.py
│   │   ├── test_rl_planner.py
│   │   └── test_localization.py
│   ├── simulation/
│   │   ├── test_gazebo_scenarios.py
│   │   └── test_randomization.py
│   └── hardware/
│       ├── test_motors.py
│       └── test_safety.py
│
├── notebooks/
│   ├── 01-rl-training.ipynb
│   ├── 02-slam-evaluation.ipynb
│   └── 03-fleet-simulation.ipynb
│
├── scripts/
│   ├── launch_slam.sh
│   ├── train_rl_policy.py
│   ├── evaluate_rl_policy.py
│   ├── run_simulation.py
│   ├── deploy_firmware.sh
│   └── calibrate_sensors.py
│
├── launch/
│   ├── robot_bringup.launch.py
│   ├── mapping.launch.py
│   ├── navigation.launch.py
│   └── rl_planner.launch.py
│
├── urdf/
│   └── warehouse_robot.urdf
│
├── worlds/
│   ├── warehouse.sdf
│   └── warehouse_small.sdf
│
└── docs/
    ├── architecture.md
    ├── safety_certification.md
    ├── deployment_guide.md
    ├── fleet_operations.md
    └── troubleshooting.md
```

### Getting Started

```bash
# 1. Copy template
cp -r TEMPLATE-AUTONOMOUS-NAVIGATION ~/my-warehouse-robot
cd ~/my-warehouse-robot

# 2. Install ROS2 and dependencies
make install-ros2

# 3. Launch simulation
ros2 launch launch/simulation.launch.py

# 4. Start SLAM mapping
ros2 launch launch/mapping.launch.py

# 5. Train RL policy
python scripts/train_rl_policy.py --steps 1000000

# 6. Evaluate in simulation
python scripts/evaluate_rl_policy.py --episodes 50

# 7. Deploy to robot
./scripts/deploy_firmware.sh --robot-id R-01
```

---

## 📚 References & Further Reading

### Academic Papers
- Hess et al. (2016) — "Real-Time Loop Closure in 2D LIDAR SLAM" — [ICRA 2016](https://doi.org/10.1109/ICRA.2016.7487258) (Cartographer)
- Schulman et al. (2017) — "Proximal Policy Optimization Algorithms" — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Tobin et al. (2017) — "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" — [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- Rösmann et al. (2017) — "Time-Optimal Nonlinear Model Predictive Control with Minimal Control Effort" (TEB Planner) — [IEEE Access](https://doi.org/10.1109/ACCESS.2017.2709252)

### ROS2 & Navigation
- ROS2 Documentation: https://docs.ros.org/en/humble/
- Nav2 (Navigation2): https://navigation.ros.org/
- SLAM Toolbox: https://github.com/SteveMacenski/slam_toolbox
- Gazebo Simulation: https://gazebosim.org/

### Hardware
- NVIDIA Jetson Orin: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
- SICK Safety LiDAR: https://www.sick.com/us/en/safety-laser-scanners
- ROS2 Real Robot: https://robots.ros.org/

---

> **Next**: [09-Recommendation-Engine.md](09-Recommendation-Engine.md) — E-commerce recommendation system with two-tower neural networks and real-time serving.
