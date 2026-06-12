# AI in Energy & Utilities

## Table of Contents
1. [Introduction](#introduction)
2. [Smart Grid Optimization](#smart-grid-optimization)
   - [Reinforcement Learning for Load Balancing](#reinforcement-learning-for-load-balancing)
   - [Grid Topology & State Estimation](#grid-topology--state-estimation)
   - [Demand Response Optimization](#demand-response-optimization)
3. [Renewable Energy Forecasting](#renewable-energy-forecasting)
   - [Solar Irradiance Prediction with ConvLSTM](#solar-irradiance-prediction-with-convlstm)
   - [Wind Speed Forecasting with Transformers](#wind-speed-forecasting-with-transformers)
   - [Ensemble Methods for Probabilistic Forecasting](#ensemble-methods-for-probabilistic-forecasting)
4. [Predictive Maintenance for Turbines](#predictive-maintenance-for-turbines)
   - [Vibration Analysis with Autoencoders](#vibration-analysis-with-autoencoders)
   - [Acoustic Emission Monitoring](#acoustic-emission-monitoring)
   - [Remaining Useful Life for Wind Turbines](#remaining-useful-life-for-wind-turbines)
5. [Energy Consumption Optimization in Buildings](#energy-consumption-optimization-in-buildings)
   - [Building Energy Model Calibration](#building-energy-model-calibration)
   - [HVAC Control with Deep RL](#hvac-control-with-deep-rl)
   - [Occupancy Prediction for Demand-Based Control](#occupancy-prediction-for-demand-based-control)
6. [Carbon Capture & Emissions Monitoring](#carbon-capture--emissions-monitoring)
   - [Satellite-Based Methane Detection](#satellite-based-methane-detection)
   - [CO2 Plume Modeling](#co2-plume-modeling)
   - [Industrial Emissions Prediction](#industrial-emissions-prediction)
7. [Nuclear Reactor Monitoring](#nuclear-reactor-monitoring)
   - [Core Temperature Prediction](#core-temperature-prediction)
   - [Anomaly Detection in Cooling Systems](#anomaly-detection-in-cooling-systems)
   - [Radiation Monitoring Networks](#radiation-monitoring-networks)
8. [Grid Anomaly Detection & Cybersecurity](#grid-anomaly-detection--cybersecurity)
   - [Fault Detection in Transmission Lines](#fault-detection-in-transmission-lines)
   - [Cyber Attack Detection for SCADA](#cyber-attack-detection-for-scada)
   - [Power Quality Disturbance Classification](#power-quality-disturbance-classification)
9. [Case Studies](#case-studies)
   - [DeepMind's Google Data Center Cooling](#deepminds-google-data-center-cooling)
   - [Octopus Energy & Smart Grid](#octopus-energy--smart-grid)
   - [Vattenfall Wind Farm Optimization](#vattenfall-wind-farm-optimization)
   - [ABB Ability for Grid Analytics](#abb-ability-for-grid-analytics)
10. [Cross-References](#cross-references)
11. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

The global energy sector is undergoing a profound transformation driven by decarbonization, decentralization, and digitization. Artificial Intelligence serves as the intelligent control layer that enables this transition — making renewable energy sources predictable, grids resilient, consumption efficient, and emissions measurable.

The AI in energy market was valued at $8.1 billion in 2023 and is projected to exceed $48 billion by 2030. This growth is fueled by several critical factors:

1. **Grid Modernization**: Aging infrastructure must accommodate distributed energy resources (solar, wind, batteries, EVs) that introduce unprecedented complexity
2. **Renewable Integration**: Solar and wind power, which are inherently variable, require accurate forecasting for grid stability
3. **Electrification**: The shift toward electric vehicles and heat pumps increases and shifts demand patterns
4. **Decarbonization Targets**: AI enables emissions monitoring, carbon capture optimization, and energy efficiency at scale
5. **Cybersecurity Threats**: Increasing digitization of grid infrastructure creates new attack surfaces that AI can help defend

Energy AI presents unique technical challenges:
- **Physics-Constrained Learning**: Models must respect physical laws (conservation of energy, Kirchhoff's laws, thermodynamics)
- **Extremely Long Time Horizons**: From microsecond power quality events to multi-decade infrastructure planning
- **Safety-Critical Control**: Grid failures can cascade into blackouts affecting millions
- **Distribution Shift**: Climate change alters weather patterns, invalidating historical data for renewable forecasting
- **Regulated Environment**: Energy markets are heavily regulated, requiring model explainability and auditability

This document provides a deep technical examination of the architectures, algorithms, and deployment patterns that power modern energy AI systems across generation, transmission, distribution, and consumption.

---

## Smart Grid Optimization

The smart grid is an electricity network that uses digital communication technology to detect and react to local changes in usage. AI is the cognitive layer that optimizes this increasingly complex system.

### Reinforcement Learning for Load Balancing

Reinforcement Learning (RL) is particularly well-suited for grid optimization because it learns optimal sequential decisions under uncertainty — exactly the problem of balancing supply and demand across a network with stochastic renewables and flexible loads.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn

class PowerGridEnv(gym.Env):
    """
    RL environment for smart grid load balancing.
    
    State: Current load, renewable generation, battery state of charge,
           grid frequency, time of day, day of week, weather forecast
    Action: Battery charge/discharge rate, flexible load curtailment,
            emergency backup dispatch
    Reward: -frequency_deviation - operating_cost - emissions_penalty
    """
    
    def __init__(self, 
                 num_buses: int = 14,  # IEEE 14-bus system
                 battery_capacity: float = 100.0,  # MWh
                 battery_power_max: float = 50.0,  # MW
                 renewable_capacity: float = 200.0,  # MW
                 time_horizon: int = 96):  # 24 hours in 15-min steps
        super().__init__()
        
        self.num_buses = num_buses
        self.time_horizon = time_horizon
        self.current_step = 0
        
        # Grid parameters
        self.battery_capacity = battery_capacity
        self.battery_power_max = battery_power_max
        self.renewable_capacity = renewable_capacity
        self.load_profile = self._generate_load_profile()
        self.renewable_profile = self._generate_renewable_profile()
        
        # State: [load_delta (num_buses), renewable_gen, battery_soc,
        #         frequency, time_of_day, day_of_week]
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * num_buses + [0, 0, 47, 0, -5]),
            high=np.array([1.0] * num_buses + [1, 1, 53, 6, 5]),
            dtype=np.float32
        )
        
        # Action: [battery_power (-1 to 1), load_curtailment (0 to 1),
        #          backup_dispatch (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.state = None
        self.reset()
    
    def _generate_load_profile(self):
        """Generate realistic load profile with seasonality."""
        np.random.seed(42)
        t = np.arange(self.time_horizon)
        
        # Daily pattern (peak at 5 PM)
        daily = 0.3 + 0.5 * (
            (t % 96) / 96  # time of day
        )
        # Weekly pattern (lower on weekends)
        weekly = 1.0 - 0.1 * (t // 96 % 7 >= 5).astype(float)
        
        base_load = (daily * weekly * 1000)  # Base load ~1000 MW
        # Add bus-level variation
        loads = base_load[:, np.newaxis] * (0.8 + 0.4 * np.random.rand(self.num_buses))
        
        return loads  # (time_horizon, num_buses)
    
    def _generate_renewable_profile(self):
        """Generate realistic renewable generation pattern."""
        t = np.arange(self.time_horizon)
        # Solar: peaks at noon (t=24 -> 6 AM, t=48 -> noon)
        solar = np.maximum(0, np.sin(np.pi * (t % 96 - 24) / 48))
        solar = solar * (0.6 + 0.4 * np.random.random(self.time_horizon))  # Cloud cover
        
        # Wind: random with autocorrelation
        wind = 0.5 + 0.5 * np.sin(2 * np.pi * t / 192 + 0.3 * np.random.randn())
        wind = np.clip(wind, 0, 1)
        
        renewable = np.column_stack([solar, wind]) * self.renewable_capacity
        
        return renewable  # (time_horizon, 2)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_soc = 0.5  # 50% initial SOC
        self.cumulative_cost = 0
        self.cumulative_emissions = 0
        
        self.state = self._get_state()
        return self.state, {}
    
    def _get_state(self):
        """Construct current observation."""
        load_delta = (self.load_profile[self.current_step] - 
                      np.mean(self.load_profile, axis=0)) / np.std(self.load_profile, axis=0)
        
        renewable_gen = (self.renewable_profile[self.current_step].sum() / 
                        (self.renewable_capacity * 2))
        
        frequency = 50.0 + np.random.randn() * 0.1  # Hz, with noise
        
        return np.concatenate([
            load_delta,
            [renewable_gen, self.battery_soc, 
             frequency, 
             (self.current_step % 96) / 96,  # Time of day (0-1)
             (self.current_step // 96) % 7 / 7]  # Day of week (0-1)
        ]).astype(np.float32)
    
    def step(self, action):
        """Execute one grid control step."""
        battery_power = action[0] * self.battery_power_max  # MW
        load_curtailment = action[1] * self.load_profile[self.current_step].sum()
        backup_dispatch = action[2] * 100  # MW
        
        # Apply battery
        self.battery_soc = np.clip(
            self.battery_soc - battery_power / self.battery_capacity * 0.25,  # 15-min step
            0, 1
        )
        
        # Compute load balance
        total_load = self.load_profile[self.current_step].sum() - load_curtailment
        total_gen = (self.renewable_profile[self.current_step].sum() + 
                    battery_power + backup_dispatch)
        
        net_load = total_load - total_gen
        frequency_deviation = net_load / 1000.0  # Hz deviation
        
        # Costs
        battery_cost = abs(battery_power) * 0.02  # $/MWh
        curtailment_cost = load_curtailment * 100  # $/MWh (high cost)
        backup_cost = backup_dispatch * 150  # $/MWh (expensive peaker plant)
        emissions = backup_dispatch * 0.5  # tons CO2/MWh (natural gas)
        
        # Reward
        frequency_penalty = -100 * abs(frequency_deviation)
        cost_penalty = -(battery_cost + curtailment_cost + backup_cost) * 0.01
        emissions_penalty = -emissions * 0.1
        
        reward = frequency_penalty + cost_penalty + emissions_penalty
        
        # Update state
        self.cumulative_cost += (battery_cost + curtailment_cost + backup_cost)
        self.cumulative_emissions += emissions
        self.current_step += 1
        
        terminated = self.current_step >= self.time_horizon
        truncated = False
        
        self.state = self._get_state()
        
        info = {
            'battery_soc': self.battery_soc,
            'frequency_deviation': frequency_deviation,
            'net_load': net_load,
            'cumulative_cost': self.cumulative_cost,
            'cumulative_emissions': self.cumulative_emissions
        }
        
        return self.state, reward, terminated, truncated, info


def train_grid_agent():
    """Train a PPO agent for smart grid load balancing."""
    env = DummyVecEnv([lambda: PowerGridEnv(num_buses=14)])
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),
            'activation_fn': nn.Tanh
        },
        verbose=1
    )
    
    model.learn(total_timesteps=1_000_000)
    model.save('smart_grid_ppo')
    
    # Evaluate
    avg_reward = evaluate_grid_policy(model, env)
    print(f"Average reward: {avg_reward}")
    
    return model


def evaluate_grid_policy(model, env, n_episodes: int = 10):
    """Evaluate trained grid policy."""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
        rewards.append(episode_reward)
    return np.mean(rewards)
```

### Grid Topology & State Estimation

Power system state estimation is the process of estimating the voltage magnitude and phase angle at each bus in the power network:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsInformedStateEstimator(nn.Module):
    """
    Physics-informed neural network for power system state estimation.
    
    Estimates bus voltages and phase angles from sparse measurements,
    with a loss function that enforces Kirchhoff's laws.
    """
    
    def __init__(self, 
                 num_buses: int = 14,
                 num_branches: int = 20,
                 measurement_dim: int = 50):
        super().__init__()
        
        self.num_buses = num_buses
        self.num_branches = num_branches
        
        # Encoder: measurements -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(measurement_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Bus parameter decoder (voltage magnitude, phase angle)
        self.v_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_buses)  # Voltage magnitudes
        )
        
        self.theta_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_buses)  # Phase angles
        )
        
        # Learnable admittance matrix (simplified)
        self.admittance = nn.Parameter(
            torch.randn(num_branches, 2) * 0.1
        )
        # bus_to_branch incidence matrix (fixed, known)
        self.register_buffer('incidence', self._build_incidence())
    
    def _build_incidence(self):
        """Build bus-branch incidence matrix A."""
        # Simplified for IEEE 14-bus system
        A = torch.zeros(self.num_branches, self.num_buses)
        # Bus connections would be defined by topology
        return A
    
    def forward(self, measurements: torch.Tensor) -> dict:
        """
        measurements: (B, M) — SCADA/PMU measurements
        """
        latent = self.encoder(measurements)
        
        # Voltage magnitudes (should be near 1.0 p.u.)
        v_mag = 0.9 + 0.2 * torch.sigmoid(self.v_decoder(latent))
        
        # Phase angles (small, typically < 30 degrees)
        theta = 0.5 * torch.tanh(self.theta_decoder(latent))
        
        return {
            'voltage_magnitude': v_mag,
            'phase_angle': theta
        }
    
    def physics_loss(self, 
                      state: dict,
                      measurements: torch.Tensor,
                      load_demands: torch.Tensor,
                      generator_outputs: torch.Tensor) -> torch.Tensor:
        """
        Physics-informed loss enforcing power flow equations.
        
        P_i = V_i * sum_j(V_j * (G_ij * cos(theta_ij) + B_ij * sin(theta_ij)))
        Q_i = V_i * sum_j(V_j * (G_ij * sin(theta_ij) - B_ij * cos(theta_ij)))
        """
        V = state['voltage_magnitude']
        theta = state['phase_angle']
        
        # Simplified power flow equations
        # Actual implementation would use full Jacobian
        
        # Power balance at each bus
        # sum(P_injected) - sum(P_load) = 0 at each bus
        
        balance_loss = 0
        for i in range(self.num_buses):
            # Compute power flow from connected branches
            P_injected = 0
            for branch_idx in range(self.num_branches):
                if self.incidence[branch_idx, i] != 0:
                    j = (self.incidence[branch_idx] != 0).nonzero(as_tuple=True)[0]
                    j = j[j != i].item()
                    
                    G = self.admittance[branch_idx, 0]
                    B = self.admittance[branch_idx, 1]
                    theta_ij = theta[0, i] - theta[0, j]
                    
                    P_flow = V[0, i] * V[0, j] * (G * torch.cos(theta_ij) + B * torch.sin(theta_ij))
                    P_injected += P_flow
            
            # Net injection should equal generation minus load
            P_net = generator_outputs[0, i] - load_demands[0, i]
            balance_loss += (P_injected - P_net) ** 2
        
        return balance_loss.mean()
```

### Demand Response Optimization

Demand Response (DR) programs incentivize consumers to shift or reduce their electricity usage during peak periods:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment

class DemandResponseOptimizer:
    """
    AI-powered demand response optimization for utility-scale programs.
    
    Matches load reduction requests with customer flexibility,
    considering customer preferences, load characteristics, and
    grid constraints.
    """
    
    def __init__(self, customer_data: dict):
        """
        customer_data: {customer_id: {
            'base_load': np.ndarray,  # Hourly load profile
            'flexible_load': float,   # kW available for DR
            'max_shift': int,         # Max hours load can be shifted
            'preference_weight': float,  # Willingness to participate (0-1)
            'critical_load': float,   # kW that cannot be interrupted
        }}
        """
        self.customers = customer_data
        self.flexibility_model = self._train_flexibility_predictor()
    
    def _train_flexibility_predictor(self):
        """Predict customer flexibility based on historical DR participation."""
        return RandomForestRegressor(n_estimators=100, max_depth=10)
    
    def optimize_dispatch(self, 
                           load_reduction_target: float,  # MW
                           time_horizon: int,  # Hours
                           grid_constraints: dict) -> dict:
        """
        Optimize which customers to dispatch and when.
        
        Returns optimal DR dispatch schedule minimizing cost
        while meeting grid constraints.
        """
        n_customers = len(self.customers)
        customer_ids = list(self.customers.keys())
        
        # Build cost matrix for assignment problem
        cost_matrix = np.zeros((n_customers, time_horizon))
        flexibility_matrix = np.zeros((n_customers, time_horizon))
        
        for i, cid in enumerate(customer_ids):
            c = self.customers[cid]
            for t in range(time_horizon):
                # Cost includes: incentive payment + inconvenience cost
                incentive_cost = c['flexible_load'] * 0.15  # $/kWh incentive
                inconvenience_cost = (1 - c['preference_weight']) * 50
                cost_matrix[i, t] = incentive_cost + inconvenience_cost
                
                # Available flexibility at time t
                flexibility_matrix[i, t] = c['flexible_load'] * \
                    self._estimate_flexibility_at_time(cid, t)
        
        # Solve as assignment problem with capacity constraints
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Build dispatch schedule
        schedule = {}
        total_reduction = 0
        total_cost = 0
        
        for i, j in zip(row_ind, col_ind):
            cid = customer_ids[i]
            reduction = min(
                flexibility_matrix[i, j],
                load_reduction_target * 1000 - total_reduction  # Convert to kW
            )
            
            if reduction > 0:
                schedule[cid] = {
                    'dispatch_time': j,
                    'reduction_kw': reduction,
                    'cost': cost_matrix[i, j],
                    'duration': self.customers[cid].get('max_duration', 1)
                }
                total_reduction += reduction
                total_cost += cost_matrix[i, j]
        
        return {
            'schedule': schedule,
            'total_reduction_mw': total_reduction / 1000,
            'total_cost': total_cost,
            'cost_per_mwh': total_cost / (total_reduction / 1000) if total_reduction > 0 else 0,
            'customers_dispatched': len(schedule)
        }
    
    def _estimate_flexibility_at_time(self, customer_id: str, hour: int) -> float:
        """Estimate customer's available flexibility at a given hour."""
        c = self.customers[customer_id]
        base = c['base_load'][hour % 24]
        
        # Less flexibility during business hours for commercial customers
        hour_factor = 0.5 if 8 <= (hour % 24) < 18 else 1.0
        
        # Apply preference weight
        return c['preference_weight'] * hour_factor
```

---

## Renewable Energy Forecasting

Accurate forecasting of solar and wind generation is critical for grid stability. Every percentage point improvement in forecast accuracy translates to millions of dollars in grid operation savings.

### Solar Irradiance Prediction with ConvLSTM

Solar forecasting combines satellite imagery with numerical weather prediction (NWP) and ground sensor data:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class ConvLSTM(nn.Module):
    """
    Convolutional LSTM for spatio-temporal solar irradiance forecasting.
    
    Processes satellite cloud imagery to predict Global Horizontal
    Irradiance (GHI) at specific locations.
    """
    
    def __init__(self, 
                 input_channels: int = 7,  # 6 satellite bands + 1 time encoding
                 hidden_dim: int = 64,
                 kernel_size: int = 3,
                 num_layers: int = 2,
                 forecast_horizon: int = 6):  # 6 hours ahead
        super().__init__()
        
        self.forecast_horizon = forecast_horizon
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stacked ConvLSTM layers
        self.conv_lstm_layers = nn.ModuleList()
        self.conv_lstm_layers.append(
            ConvLSTMCell(input_channels, hidden_dim, kernel_size)
        )
        for _ in range(1, num_layers):
            self.conv_lstm_layers.append(
                ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
            )
        
        # Output convolution: hidden -> GHI prediction
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, forecast_horizon, 1)
        )
    
    def forward(self, satellite_images: torch.Tensor) -> torch.Tensor:
        """
        satellite_images: (B, T, C, H, W) — sequence of satellite images
        Returns: (B, forecast_horizon, H, W) — GHI predictions
        """
        B, T, C, H, W = satellite_images.shape
        
        # Initialize states
        h_t = [torch.zeros(B, self.hidden_dim, H, W, device=satellite_images.device)
               for _ in range(self.num_layers)]
        c_t = [torch.zeros(B, self.hidden_dim, H, W, device=satellite_images.device)
               for _ in range(self.num_layers)]
        
        # Process temporal sequence
        for t in range(T):
            x_t = satellite_images[:, t, :, :, :]
            
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.conv_lstm_layers[layer](
                    x_t if layer == 0 else h_t[layer-1],
                    (h_t[layer], c_t[layer])
                )
        
        # Generate forecast
        forecast = self.output_conv(h_t[-1])
        
        return forecast


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatio-temporal modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding
        )
    
    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat([x, h_prev], dim=1)
        
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c


class SolarForecastSystem:
    """
    End-to-end solar irradiance forecasting system combining
    satellite imagery, NWP, and ground measurements.
    """
    
    def __init__(self, 
                 model_path: str = 'solar_convlstm.pth',
                 location: Tuple[float, float] = (37.7749, -122.4194)):  # San Francisco
        self.location = location
        self.model = ConvLSTM(forecast_horizon=6)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Clear-sky model (physical baseline)
        self.clear_sky_model = self._init_clear_sky_model()
    
    def _init_clear_sky_model(self):
        """Initialize physical clear-sky irradiance model (Ineichen/Perez)."""
        class ClearSkyModel:
            def __init__(self, lat, lon):
                self.lat = lat
                self.lon = lon
            
            def compute_ghi(self, hour, day_of_year):
                """Simplified clear-sky GHI computation."""
                # Solar zenith angle
                declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
                hour_angle = 15 * (hour - 12)
                
                zenith = np.radians(self.lat - declination)
                cos_zenith = np.cos(np.radians(abs(hour_angle))) * np.cos(np.radians(self.lat)) * \
                             np.cos(np.radians(declination)) + np.sin(np.radians(self.lat)) * \
                             np.sin(np.radians(declination))
                
                if cos_zenith < 0:
                    return 0
                
                # Extraterrestrial irradiance
                G_sc = 1361  # Solar constant W/m²
                G_o = G_sc * (1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365)))
                
                # Clear-sky transmittance (Linke turbidity = 3)
                TL = 3
                GHI_clear = G_o * cos_zenith * np.exp(-0.001 * TL / cos_zenith)
                
                return max(0, GHI_clear)
        
        return ClearSkyModel(*self.location)
    
    def predict(self, 
                 satellite_images: np.ndarray,
                 nwp_data: np.ndarray,
                 ground_measurements: np.ndarray,
                 timestamp: int) -> dict:
        """
        Predict solar irradiance for the next 6 hours.
        
        Returns both ML forecast and clear-sky baseline.
        """
        # ML forecast
        with torch.no_grad():
            sat_tensor = torch.FloatTensor(satellite_images).unsqueeze(0)
            ml_forecast = self.model(sat_tensor).squeeze().numpy()
        
        # Clear-sky baseline
        day_of_year = timestamp // 86400 % 365
        clear_sky = np.array([
            self.clear_sky_model.compute_ghi(t, day_of_year)
            for t in range(6)
        ])
        
        # Blend ML forecast with clear-sky model (physical constraints)
        # Ensure forecast doesn't exceed clear-sky limits
        final_forecast = np.minimum(ml_forecast, clear_sky * 1.1)
        
        return {
            'forecast_ghi_wm2': final_forecast.tolist(),
            'clear_sky_ghi_wm2': clear_sky.tolist(),
            'confidence': self._estimate_uncertainty(ml_forecast, clear_sky),
            'forecast_hours': [timestamp + i * 3600 for i in range(6)]
        }
    
    def _estimate_uncertainty(self, ml_forecast: np.ndarray, 
                               clear_sky: np.ndarray) -> float:
        """Estimate prediction uncertainty based on cloud variability."""
        diff = np.abs(ml_forecast - clear_sky)
        cloud_cover_ratio = diff / (clear_sky + 1e-10)
        return float(1.0 - np.clip(np.mean(cloud_cover_ratio), 0, 0.8))
```

### Wind Speed Forecasting with Transformers

Wind forecasting requires modeling complex atmospheric dynamics across multiple spatial and temporal scales:

```python
import torch
import torch.nn as nn
import math

class WindSpeedTransformer(nn.Module):
    """
    Transformer-based wind speed and power forecasting.
    
    Input: NWP data (wind speed, direction, temperature, pressure, humidity)
           at multiple heights and grid points around the wind farm
    Output: Wind speed and power for each turbine, 1-72 hours ahead
    """
    
    def __init__(self, 
                 nwp_features: int = 10,
                 n_grid_points: int = 100,
                 n_turbines: int = 50,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 forecast_horizon: int = 72):  # 72 hours
        super().__init__()
        
        self.n_grid_points = n_grid_points
        self.n_turbines = n_turbines
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(nwp_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Spatial encoding (learned grid position embeddings)
        self.spatial_embed = nn.Embedding(n_grid_points, d_model // 2)
        self.turbine_embed = nn.Embedding(n_turbines, d_model // 2)
        
        # Transformer encoder (NWP grid -> latent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder (latent -> turbine forecasts)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Turbine query embeddings
        self.turbine_queries = nn.Parameter(
            torch.randn(n_turbines * forecast_horizon, d_model)
        )
        
        # Output heads
        self.wind_speed_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.wind_power_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Wind direction (circular output)
        self.wind_dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # sin(theta), cos(theta)
        )
    
    def forward(self, nwp_data: torch.Tensor) -> dict:
        """
        nwp_data: (B, T_in, N_grid, F) — NWP time series
        """
        B, T, N, F = nwp_data.shape
        
        # Project and add spatial + temporal encoding
        x = self.input_proj(nwp_data)  # (B, T, N, D)
        x = x.reshape(B * N, T, -1)
        x = self.pos_encoder(x)
        
        # Add spatial embeddings
        spatial_emb = self.spatial_embed.weight.unsqueeze(0).unsqueeze(1)
        spatial_emb = spatial_emb.expand(B, T, -1, -1).reshape(B * N, T, -1)
        # Project spatial emb to d_model
        spatial_proj = nn.Linear(self.d_model // 2, x.size(-1), device=x.device)
        x = x + spatial_proj(spatial_emb[:, :, :self.d_model // 2])
        
        # Encode
        memory = self.encoder(x)  # (B*N, T, D)
        
        # Decode with turbine queries
        turbine_queries = self.turbine_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(turbine_queries, memory)
        
        # Reshape to (B, N_turbines, T_forecast, D)
        decoded = decoded.view(B, self.n_turbines, self.forecast_horizon, -1)
        
        # Predictions
        wind_speed = self.wind_speed_head(decoded).squeeze(-1)
        wind_power = self.wind_power_head(decoded).squeeze(-1)
        wind_dir = self.wind_dir_head(decoded)
        wind_dir = torch.atan2(wind_dir[..., 0], wind_dir[..., 1])  # Convert to angle
        
        return {
            'wind_speed': wind_speed,  # (B, N_turbines, T_forecast)
            'wind_power': wind_power,  # (B, N_turbines, T_forecast)
            'wind_direction': wind_dir  # (B, N_turbines, T_forecast)
        }


class ProbabilisticWindForecast(nn.Module):
    """
    Probabilistic wind power forecasting using quantile regression.
    Outputs prediction intervals for risk-aware grid operations.
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_dim: int = 256,
                 quantiles: list = [0.05, 0.25, 0.5, 0.75, 0.95]):
        super().__init__()
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_quantiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile predictions."""
        return self.network(x)
    
    def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Quantile loss (pinball loss) for probabilistic forecasting.
        
        L(y, y_hat, q) = max(q * (y - y_hat), (q-1) * (y - y_hat))
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - pred[..., i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.unsqueeze(-1))
        
        return torch.cat(losses, dim=-1).mean()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

### Ensemble Methods for Probabilistic Forecasting

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy.stats import norm

class WindPowerEnsemble:
    """
    Ensemble forecasting system combining multiple model types.
    Provides probabilistic forecasts with calibrated uncertainty.
    """
    
    def __init__(self):
        self.models = {
            'gbr': GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05
            ),
            'rf': RandomForestRegressor(
                n_estimators=200, max_depth=10
            ),
            'lstm': self._build_lstm_model(),
            'persistence': self._build_persistence_model()
        }
        
        # Meta-learner for ensemble weighting
        self.meta_learner = RandomForestRegressor(
            n_estimators=50, max_depth=3
        )
    
    def _build_lstm_model(self):
        """Lightweight LSTM for wind power prediction."""
        import torch
        return torch.nn.Sequential(
            torch.nn.LSTM(10, 64, batch_first=True),
            torch.nn.Linear(64, 1)
        )
    
    def _build_persistence_model(self):
        """Simple persistence (naive) model as baseline."""
        class PersistenceModel:
            def predict(self, x):
                return x[:, -1, 0]  # Last observed value
        
        return PersistenceModel()
    
    def predict_ensemble(self, features: np.ndarray) -> dict:
        """
        Generate ensemble predictions with uncertainty quantification.
        """
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'lstm':
                # PyTorch inference
                with torch.no_grad():
                    feat_tensor = torch.FloatTensor(features)
                    pred = model(feat_tensor).numpy()
            else:
                pred = model.predict(features)
            predictions[name] = pred
        
        # Ensemble statistics
        pred_array = np.array(list(predictions.values()))
        ensemble_mean = pred_array.mean(axis=0)
        ensemble_std = pred_array.std(axis=0)
        
        # Calibrated prediction intervals using split conformal prediction
        calibrated_intervals = self._conformal_calibration(
            ensemble_mean, ensemble_std
        )
        
        return {
            'point_forecast': ensemble_mean,
            'individual_models': predictions,
            'prediction_interval_90': calibrated_intervals['90%'],
            'prediction_interval_80': calibrated_intervals['80%'],
            'prediction_interval_50': calibrated_intervals['50%'],
            'ensemble_std': ensemble_std,
            'model_weights': self._compute_model_weights(pred_array)
        }
    
    def _conformal_calibration(self, mean: np.ndarray, 
                                std: np.ndarray) -> dict:
        """Conformal prediction intervals."""
        intervals = {}
        for confidence, z_score in [('90%', 1.645), ('80%', 1.282), ('50%', 0.674)]:
            lower = mean - z_score * std
            upper = mean + z_score * std
            intervals[confidence] = np.stack([lower, upper], axis=-1)
        return intervals
    
    def _compute_model_weights(self, predictions: np.ndarray) -> dict:
        """Compute optimal ensemble weights based on recent performance."""
        # Inverse error weighting
        errors = np.std(predictions, axis=1) + 1e-10
        weights = 1.0 / errors
        weights = weights / weights.sum()
        
        return {
            name: float(w) 
            for name, w in zip(self.models.keys(), weights)
        }
```

---

## Predictive Maintenance for Turbines

### Vibration Analysis with Autoencoders

Wind turbine gearboxes and bearings generate characteristic vibration signatures. Changes in these signatures precede failures by weeks or months:

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from typing import Dict

class TurbineVibrationAutoencoder(nn.Module):
    """
    1D Convolutional autoencoder for wind turbine vibration anomaly detection.
    
    Learns normal vibration patterns during healthy operation.
    High reconstruction error indicates developing faults.
    """
    
    def __init__(self, 
                 sensor_channels: int = 3,  # X, Y, Z accelerometers
                 sequence_length: int = 4096):
        super().__init__()
        
        self.sensor_channels = sensor_channels
        self.seq_length = sequence_length
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(sensor_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten()
        )
        
        # Bottleneck
        encoder_out_size = 256 * 32
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_out_size, 64),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder_proj = nn.Linear(64, encoder_out_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, sensor_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
    
    def forward(self, x):
        """
        x: (batch, sensor_channels, sequence_length)
        """
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded_proj = self.decoder_proj(bottleneck)
        b, c, l = x.shape
        decoded = decoded_proj.view(b, -1, 32)
        reconstructed = self.decoder(decoded)
        
        # Trim to original length
        if reconstructed.shape[-1] > self.seq_length:
            reconstructed = reconstructed[:, :, :self.seq_length]
        
        return reconstructed, bottleneck


class TurbineHealthMonitor:
    """
    End-to-end wind turbine health monitoring system.
    Combines vibration analysis, SCADA data, and oil analysis.
    """
    
    def __init__(self, turbine_id: str, model_path: str = None):
        self.turbine_id = turbine_id
        self.autoencoder = TurbineVibrationAutoencoder()
        self.baseline_stats = None
        
        if model_path:
            self.autoencoder.load_state_dict(torch.load(model_path))
        self.autoencoder.eval()
    
    def fit_baseline(self, vibration_data: np.ndarray):
        """Establish baseline statistics from healthy operation data."""
        with torch.no_grad():
            data_tensor = torch.FloatTensor(vibration_data)
            _, bottleneck = self.autoencoder(data_tensor)
            self.baseline_stats = {
                'mean': bottleneck.mean(dim=0),
                'std': bottleneck.std(dim=0),
                'reconstruction_errors': []
            }
    
    def analyze_vibration(self, 
                           vibration_data: np.ndarray,
                           sampling_rate: int = 25600) -> Dict:
        """
        Comprehensive vibration analysis for fault detection.
        
        Returns:
        - Anomaly score (0-1)
        - Identified fault type (if any)
        - Frequency domain features
        - Trend analysis
        """
        # Preprocess: band-pass filter for relevant frequency range
        sos = signal.butter(4, [5, 1000], btype='band', fs=sampling_rate, output='sos')
        filtered = signal.sosfilt(sos, vibration_data, axis=-1)
        
        # Time-domain features
        rms = np.sqrt(np.mean(filtered**2, axis=-1))
        peak_factor = np.max(np.abs(filtered), axis=-1) / (rms + 1e-10)
        kurtosis = signal.kurtosis(filtered, axis=-1)
        
        # Frequency-domain features
        f, psd = signal.welch(filtered, fs=sampling_rate, nperseg=1024)
        
        # ML-based anomaly detection
        with torch.no_grad():
            data_tensor = torch.FloatTensor(filtered).unsqueeze(0)
            reconstructed, bottleneck = self.autoencoder(data_tensor)
            
            # Reconstruction error
            mse = nn.MSELoss()(reconstructed, data_tensor).item()
            
            # Latent space anomaly
            if self.baseline_stats is not None:
                z_score = (bottleneck - self.baseline_stats['mean'].to(bottleneck.device)) / \
                          (self.baseline_stats['std'].to(bottleneck.device) + 1e-10)
                latent_anomaly = torch.abs(z_score).mean().item()
            else:
                latent_anomaly = 0
        
        # Combined anomaly score
        anomaly_score = 0.4 * mse + 0.3 * latent_anomaly + 0.3 * (kurtosis.mean() / 3 - 1)
        anomaly_score = np.clip(anomaly_score / 10, 0, 1)
        
        # Fault identification
        fault_type = self._identify_fault(
            rms, peak_factor, kurtosis, f, psd, anomaly_score
        )
        
        return {
            'anomaly_score': anomaly_score,
            'fault_type': fault_type,
            'severity': 'critical' if anomaly_score > 0.8 else \
                       'warning' if anomaly_score > 0.5 else 'normal',
            'features': {
                'rms': rms.tolist(),
                'peak_factor': peak_factor.tolist(),
                'kurtosis': kurtosis.tolist(),
                'spectral_peaks': self._find_spectral_peaks(f, psd)
            },
            'reconstruction_error': mse
        }
    
    def _identify_fault(self, rms, peak_factor, kurtosis, f, psd, 
                         anomaly_score) -> str:
        """Identify specific fault type from vibration signature."""
        # Bearing fault frequencies (simplified)
        if anomaly_score > 0.6:
            # Check for bearing fault signatures
            # BPFI, BPFO, BSF, FTF frequencies
            if kurtosis.max() > 5:
                return 'bearing_fault'
            elif peak_factor.max() > 10:
                return 'gearbox_tooth_break'
            else:
                return 'imbalance_or_misalignment'
        return 'normal'
    
    def _find_spectral_peaks(self, f: np.ndarray, psd: np.ndarray) -> list:
        """Find dominant spectral peaks for diagnostic purposes."""
        peaks = []
        for channel in range(psd.shape[0]):
            # Find peaks in PSD
            from scipy.signal import find_peaks
            peak_idx, properties = find_peaks(psd[channel], height=np.median(psd[channel]) * 3)
            for idx in peak_idx[:5]:  # Top 5 peaks
                peaks.append({
                    'frequency_hz': float(f[idx]),
                    'magnitude': float(psd[channel][idx])
                })
        return peaks


class WindFarmFleetMonitor:
    """
    Fleet-level wind turbine health monitoring across an entire wind farm.
    """
    
    def __init__(self):
        self.turbines: Dict[str, TurbineHealthMonitor] = {}
    
    def add_turbine(self, turbine_id: str, monitor: TurbineHealthMonitor):
        self.turbines[turbine_id] = monitor
    
    def scan_fleet(self) -> Dict:
        """Scan all turbines and aggregate health status."""
        fleet_health = {
            'total_turbines': len(self.turbines),
            'healthy': 0,
            'warning': 0,
            'critical': 0,
            'turbines': {}
        }
        
        for tid, monitor in self.turbines.items():
            # In production, this would process recent SCADA data
            status = {
                'severity': 'normal',
                'anomaly_score': 0.1,  # Placeholder
                'fault_type': 'none'
            }
            
            fleet_health['turbines'][tid] = status
            
            if status['severity'] == 'critical':
                fleet_health['critical'] += 1
            elif status['severity'] == 'warning':
                fleet_health['warning'] += 1
            else:
                fleet_health['healthy'] += 1
        
        # Maintenance prioritization
        fleet_health['maintenance_priority'] = self._prioritize_maintenance()
        
        # Production loss estimation
        fleet_health['estimated_production_loss_mwh'] = self._estimate_losses()
        
        return fleet_health
    
    def _prioritize_maintenance(self) -> list:
        """Prioritize turbine maintenance based on health + production loss."""
        priorities = []
        for tid, monitor in self.turbines.items():
            # Combined score: health risk * production impact
            priority_score = 0.5  # Placeholder
            priorities.append({
                'turbine_id': tid,
                'priority_score': priority_score,
                'recommended_action': 'inspect' if priority_score > 0.7 else 'monitor'
            })
        
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def _estimate_losses(self) -> float:
        """Estimate production losses due to degraded turbines."""
        total_loss = 0
        for tid, monitor in self.turbines.items():
            # Loss estimation based on health status
            total_loss += 10  # Placeholder MWh
        return total_loss
```

---

## Energy Consumption Optimization in Buildings

Buildings account for 40% of global energy consumption. AI-driven optimization can reduce this by 20-40%.

### Building Energy Model Calibration

```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict

class BuildingEnergyModel:
    """
    Physics-based building energy model with AI calibration.
    
    Uses a simplified RC (Resistor-Capacitor) thermal network model
    whose parameters are calibrated using ML techniques.
    """
    
    def __init__(self, building_params: Dict):
        """
        building_params: {
            'floor_area': float,  # m²
            'num_floors': int,
            'window_ratio': float,  # 0-1
            'construction_type': str,  # 'heavy', 'medium', 'light'
            'hvac_type': str,  # 'vav', 'chiller_boiler', 'heat_pump'
        }
        """
        self.params = building_params
        self.thermal_params = self._initialize_thermal_parameters()
    
    def _initialize_thermal_parameters(self) -> Dict:
        """Initialize RC thermal network parameters."""
        # Thermal resistance and capacitance values
        return {
            'R_wall': 3.0,      # Wall thermal resistance (m²K/W)
            'R_window': 0.5,    # Window thermal resistance
            'R_roof': 4.0,      # Roof thermal resistance
            'C_air': 10000,     # Air thermal capacitance (J/K)
            'C_mass': 500000,   # Building mass thermal capacitance
            'infiltration': 0.5 # Air changes per hour
        }
    
    def simulate(self, 
                 weather_data: np.ndarray,
                 hvac_schedule: np.ndarray,
                 internal_gains: np.ndarray) -> np.ndarray:
        """
        Simulate building energy consumption.
        
        weather_data: (T, 3) — outdoor temperature, solar radiation, humidity
        hvac_schedule: (T, 2) — cooling setpoint, heating setpoint
        internal_gains: (T, 3) — occupancy gains, equipment gains, lighting gains
        
        Returns: (T,) — total power consumption (kW)
        """
        T = len(weather_data)
        power = np.zeros(T)
        
        # State variables
        T_air = 22.0  # Initial indoor temperature (°C)
        T_mass = 22.0  # Initial mass temperature
        
        dt = 1.0  # Time step (hours)
        
        for t in range(T):
            T_out = weather_data[t, 0]
            solar = weather_data[t, 1]
            q_occ, q_eq, q_light = internal_gains[t]
            
            # Heat balance
            # Conduction through envelope
            total_UA = (self.params['floor_area'] * self.params['num_floors'] / 
                       (self.thermal_params['R_wall'] + 
                        self.thermal_params['R_window'] * self.params['window_ratio'] +
                        self.thermal_params['R_roof']))
            
            q_cond = total_UA * (T_out - T_air)
            
            # Solar heat gain
            q_solar = solar * self.params['window_ratio'] * 0.5 * self.params['floor_area']
            
            # Infiltration
            q_inf = self.thermal_params['infiltration'] * self.params['floor_area'] * \
                    3.0 * (T_out - T_air) / 3600
            
            # Internal gains
            q_int = q_occ + q_eq + q_light
            
            # HVAC
            T_cool_set = hvac_schedule[t, 0]
            T_heat_set = hvac_schedule[t, 1]
            
            q_hvac = 0
            if T_air > T_cool_set:
                q_hvac = - (T_air - T_cool_set) * 5000  # Cooling
            elif T_air < T_heat_set:
                q_hvac = (T_heat_set - T_air) * 5000  # Heating
            
            # Update temperatures
            T_air += dt / self.thermal_params['C_air'] * (
                q_cond + q_inf + q_solar + q_int + q_hvac +
                (T_mass - T_air) * 100  # Coupling to thermal mass
            )
            
            T_mass += dt / self.thermal_params['C_mass'] * (
                (T_air - T_mass) * 100
            )
            
            # Power consumption (simplified COP)
            if q_hvac > 0:
                cop = 3.5  # Heat pump
            else:
                cop = 4.0  # Cooling
            
            power[t] = abs(q_hvac) / cop / 1000  # kW
        
        return power


class MLCalibratedEnergyModel:
    """
    Hybrid physics-ML building energy model.
    Neural network corrects physics model biases using actual data.
    """
    
    def __init__(self, physics_model: BuildingEnergyModel):
        self.physics = physics_model
        self.correction_model = self._build_correction_model()
    
    def _build_correction_model(self):
        """Neural network to learn physics model correction."""
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(8, 64),  # weather + schedule + internal gains features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)  # Power correction (kW)
        )
    
    def calibrate(self, 
                  historical_weather: np.ndarray,
                  historical_schedule: np.ndarray,
                  historical_gains: np.ndarray,
                  actual_power: np.ndarray):
        """Calibrate correction model using historical data."""
        # Physics simulation
        sim_power = self.physics.simulate(
            historical_weather, historical_schedule, historical_gains
        )
        
        # Error = actual - physics
        error = actual_power - sim_power
        
        # Train correction model
        features = np.concatenate([
            historical_weather,
            historical_schedule,
            historical_gains
        ], axis=1)
        
        import torch
        import torch.optim as optim
        
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(error)
        
        optimizer = optim.Adam(self.correction_model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            optimizer.zero_grad()
            pred = self.correction_model(X).squeeze()
            loss = torch.nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
    
    def predict(self, weather: np.ndarray, schedule: np.ndarray, 
                gains: np.ndarray) -> Dict:
        """Predict building energy consumption with ML correction."""
        # Physics simulation
        physics_power = self.physics.simulate(weather, schedule, gains)
        
        # ML correction
        features = np.concatenate([weather, schedule, gains], axis=1)
        import torch
        with torch.no_grad():
            correction = self.correction_model(
                torch.FloatTensor(features)
            ).numpy().squeeze()
        
        final_power = physics_power + correction
        
        return {
            'total_power_kw': final_power.tolist(),
            'physics_baseline_kw': physics_power.tolist(),
            'ml_correction_kw': correction.tolist(),
            'peak_demand_kw': float(np.max(final_power)),
            'total_energy_kwh': float(np.sum(final_power)),
            'savings_potential': self._estimate_savings(final_power)
        }
    
    def _estimate_savings(self, current_power: np.ndarray) -> Dict:
        """Estimate potential energy savings through optimization."""
        # Baseline: current operation
        baseline = np.sum(current_power)
        
        # Optimal: shift to off-peak + reduce peak
        optimal = baseline * 0.85  # Simplified: 15% savings potential
        
        return {
            'baseline_kwh': float(baseline),
            'optimal_kwh': float(optimal),
            'savings_kwh': float(baseline - optimal),
            'savings_percent': 15.0,
            'annual_savings_usd': float((baseline - optimal) * 0.12 * 365)
        }
```

### HVAC Control with Deep RL

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

class HVACControlEnv(gym.Env):
    """
    Reinforcement learning environment for HVAC optimization.
    
    Objective: Minimize energy consumption while maintaining comfort.
    
    State: Indoor temp, outdoor temp, occupancy, time of day,
           zone temperatures (multi-zone), humidity, CO2
    Action: Supply air temperature setpoint, fan speed, damper positions
    Reward: -energy_cost - comfort_penalty
    """
    
    def __init__(self, 
                 num_zones: int = 5,
                 floor_area: float = 1000.0,  # m²
                 max_steps: int = 96):  # 24 hours at 15-min intervals
        super().__init__()
        
        self.num_zones = num_zones
        self.floor_area = floor_area
        self.max_steps = max_steps
        self.current_step = 0
        
        # Observation: [T_zone (num_zones), T_out, occupancy (num_zones),
        #               hour, day_of_week, humidity, CO2, solar]
        obs_dim = num_zones * 3 + 5
        self.observation_space = spaces.Box(
            low=np.array([10] * num_zones + [-10] + [0] * num_zones + [0, 0, 0, 0, 0]),
            high=np.array([40] * num_zones + [50] + [1] * num_zones + [23, 6, 100, 2000, 1000]),
            dtype=np.float32
        )
        
        # Action: [supply_temp (16-30°C), fan_speed (0-1), 
        #           damper_positions (num_zones, 0-1)]
        self.action_space = spaces.Box(
            low=np.array([16] + [0] + [0] * num_zones),
            high=np.array([30] + [1] + [1] * num_zones),
            dtype=np.float32
        )
        
        # Building thermal dynamics (simplified)
        self.zone_temps = np.ones(num_zones) * 22.0
        self.occupancy = np.zeros(num_zones)
        self.weather_profile = self._generate_weather()
    
    def _generate_weather(self):
        """Generate synthetic weather profile."""
        t = np.arange(self.max_steps)
        # Temperature: sinusoidal with peak at 2 PM
        temp = 25 + 10 * np.sin(2 * np.pi * (t - 8) / 96)
        # Solar: peaks at noon
        solar = 600 * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
        return np.column_stack([temp, solar])
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.zone_temps = np.ones(self.num_zones) * 22.0
        self.occupancy = np.random.randint(0, 2, size=self.num_zones)
        self.energy_consumed = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.concatenate([
            self.zone_temps,
            [self.weather_profile[self.current_step, 0]],
            self.occupancy,
            [self.current_step % 96,  # Hour in 15-min steps
             self.current_step // 96 % 7,
             np.random.normal(50, 5),  # Humidity
             np.random.normal(400, 50),  # CO2
             self.weather_profile[self.current_step, 1]]  # Solar
        ]).astype(np.float32)
    
    def step(self, action):
        supply_temp = action[0]
        fan_speed = np.clip(action[1], 0, 1)
        damper_pos = np.clip(action[2:], 0, 1)
        
        T_out = self.weather_profile[self.current_step, 0]
        solar = self.weather_profile[self.current_step, 1]
        
        # Zone thermal dynamics
        for zone in range(self.num_zones):
            # Heat transfer
            q_envelope = 0.1 * (T_out - self.zone_temps[zone])
            q_solar = solar * 0.01 * damper_pos[zone]
            q_occupancy = self.occupancy[zone] * 100  # 100W per person
            q_hvac = fan_speed * damper_pos[zone] * 5000 * (supply_temp - self.zone_temps[zone])
            
            # Temperature update (simplified)
            self.zone_temps[zone] += (q_envelope + q_solar + q_occupancy + q_hvac) / 50000
            self.zone_temps[zone] = np.clip(self.zone_temps[zone], 16, 35)
        
        # Energy consumption
        energy = (fan_speed * 10 +  # Fan power
                  abs(supply_temp - T_out) * fan_speed * 0.5)  # Heating/cooling
        self.energy_consumed += energy
        
        # Comfort penalty (PMV-based simplified)
        comfort_penalty = 0
        for zone in range(self.num_zones):
            if self.occupancy[zone]:
                temp_deviation = abs(self.zone_temps[zone] - 22.0)
                comfort_penalty += temp_deviation ** 2 * 10
        
        # Reward
        energy_cost = energy * 0.15  # $/kWh
        reward = -(energy_cost + comfort_penalty * 0.01)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'energy_kwh': self.energy_consumed,
            'avg_comfort_deviation': np.mean(abs(self.zone_temps - 22.0))
        }
        
        return self._get_obs(), reward, terminated, truncated, info


def train_hvac_agent():
    """Train SAC agent for HVAC control."""
    env = DummyVecEnv([lambda: HVACControlEnv(num_zones=5)])
    
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={
            'net_arch': dict(pi=[256, 256], qf=[256, 256]),
            'activation_fn': nn.ReLU
        },
        verbose=1
    )
    
    model.learn(total_timesteps=500000)
    model.save('hvac_sac')
    
    return model
```

---

## Grid Anomaly Detection

### Power Quality Disturbance Classification

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import signal

class PowerQualityClassifier(nn.Module):
    """
    1D CNN for power quality disturbance classification.
    
    Classes:
    0: Normal
    1: Sag (voltage dip)
    2: Swell (voltage surge)
    3: Interruption
    4: Harmonics
    5: Transient
    6: Flicker
    7: Notch
    """
    
    def __init__(self, input_channels: int = 3,  # V_a, V_b, V_c
                 sequence_length: int = 2560):  # 10 cycles at 256 samples/cycle
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 8)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) voltage waveforms
        """
        features = self.features(x)
        return self.classifier(features)
    
    def extract_features(self, voltage_signal: np.ndarray, 
                          sampling_rate: int = 12800) -> dict:
        """Extract power quality features for interpretability."""
        # RMS calculation
        rms = np.sqrt(np.mean(voltage_signal**2, axis=-1))
        
        # THD (Total Harmonic Distortion)
        f, psd = signal.welch(voltage_signal, fs=sampling_rate, nperseg=1024)
        fundamental_idx = np.argmin(np.abs(f - 60))  # 60 Hz fundamental
        fundamental_power = psd[:, fundamental_idx]
        harmonic_power = np.sum(psd[:, fundamental_idx*2:fundamental_idx*10], axis=1)
        thd = np.sqrt(harmonic_power / (fundamental_power + 1e-10))
        
        # Voltage imbalance
        v_mag = np.abs(np.fft.fft(voltage_signal, axis=-1)[:, fundamental_idx])
        imbalance = (np.max(v_mag) - np.min(v_mag)) / np.mean(v_mag) * 100
        
        # Flicker (IEC 61000-4-15 simplified)
        envelope = np.abs(signal.hilbert(voltage_signal[0]))
        flicker = np.std(envelope) / np.mean(envelope)
        
        return {
            'rms_voltage': rms.tolist(),
            'thd_percent': (thd * 100).tolist(),
            'voltage_imbalance_percent': float(imbalance),
            'flicker_intensity': float(flicker),
            'sag_detected': bool(np.any(rms < 0.9)),
            'swell_detected': bool(np.any(rms > 1.1))
        }
```

---

## Case Studies

### DeepMind's Google Data Center Cooling

In 2016, DeepMind applied reinforcement learning to reduce Google's data center cooling energy by 40% — one of the most celebrated applications of AI in energy.

**Technical Approach**:

1. **Data Collection**: Historical data from thousands of sensors (temperatures, pressures, pump speeds, valve positions, chiller loads)
2. **Model Architecture**: 
   - **Neural network ensemble predicting PUE (Power Usage Effectiveness)**
   - **RL agent (Deep Q-Network variant) optimizing cooling actions**
3. **Constraints**: Safety constraints hard-coded to prevent actions outside operating limits
4. **Evaluation**: A/B testing against human operators, running in "recommendation mode" before full automation

```python
class DataCenterCoolingRL:
    """
    Simplified version of DeepMind's data center cooling optimization.
    """
    
    def __init__(self, num_it_rooms: int = 10):
        self.num_rooms = num_it_rooms
        
        # State: [IT_load, outdoor_temp, cold_aisle_temp, 
        #         hot_aisle_temp, chiller_temp, pump_speed, 
        #         fan_speed, humidity, PUE]
        self.state_dim = num_it_rooms * 5 + 4
        
        # Action: chiller setpoint, pump speed, fan speed, 
        #         cooling tower setup, valve positions
        self.action_dim = num_it_rooms + 4
        
        self.model = self._build_dqn()
    
    def _build_dqn(self):
        """Deep Q-Network for cooling optimization."""
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.action_dim)
        )
    
    def optimize(self, current_state: np.ndarray) -> dict:
        """Generate optimal cooling setpoints."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        
        return {
            'recommended_action': action,
            'q_values': q_values.numpy().tolist(),
            'expected_improvement': float(q_values.max() - q_values.mean())
        }
```

### Octopus Energy & Smart Grid

Octopus Energy uses AI for dynamic pricing, demand forecasting, and smart charging of electric vehicles.

**Key Systems**:
1. **Agile Octopus**: Half-hourly wholesale price tracking with ML-based price prediction
2. **Octopus Go**: Smart EV charging scheduling using RL
3. **Kraken Platform**: AI-powered customer service and grid management

### Vattenfall Wind Farm Optimization

Vattenfall uses AI across its 2,000+ turbine fleet for:
1. **Yaw Optimization**: RL for optimal turbine alignment to wind direction
2. **Wake Steering**: Coordinating upstream turbines to reduce wake effects on downstream turbines
3. **Ice Detection**: Vibration analysis for blade icing detection
4. **Bat and Bird Detection**: Computer vision-based curtailment systems

### ABB Ability for Grid Analytics

ABB's grid analytics platform uses AI for:
1. **Transformer Health**: Dissolved gas analysis with ML for early fault detection
2. **Line Rating**: Dynamic thermal rating using weather-adaptive ML models that increase line capacity 10-30%
3. **Substation Automation**: Pattern recognition for equipment degradation
4. **Fault Location**: Wavelet-based analysis + ML for precise fault localization

---

## Cross-References

This document intersects with several other domains in the AI Applications series:

- **[04-Manufacturing-AI.md](04-Manufacturing-AI.md)**: Predictive maintenance for wind turbines shares identical autoencoder and RUL architectures with manufacturing equipment monitoring. Digital twins in factories parallel building energy models.

- **[09-Transportation-AI.md](09-Transportation-AI.md)**: EV fleet charging optimization directly integrates with smart grid load balancing. Battery health monitoring for EVs uses the same techniques as grid storage monitoring.

- **[03-Finance-AI.md](03-Finance-AI.md)**: Energy trading and risk management use the same time-series forecasting models. Carbon credit markets use blockchain + AI tracking.

- **[08-Agriculture-AI.md](08-Agriculture-AI.md)**: Solar forecasting techniques are shared between agricultural (evapotranspiration modeling) and grid applications. Biogas and biomass energy from agriculture integrates with renewable energy forecasting.

- **[06-Retail-AI.md](06-Retail-AI.md)**: Demand forecasting for retail uses the same architectures (Transformer, LSTM) as renewable generation forecasting. Building energy management shares occupancy prediction models with retail footfall analysis.

---

## Summary & Conclusion

This document has provided a deep technical exploration of AI applications in the energy sector, covering:

1. **Smart Grid Optimization**: Reinforcement Learning (PPO, SAC) for load balancing, physics-informed neural networks for state estimation, demand response optimization with combinatorial assignment.

2. **Renewable Energy Forecasting**: ConvLSTM for solar irradiance prediction from satellite imagery, Transformer models for wind speed/power forecasting, ensemble methods with conformal prediction for probabilistic forecasting.

3. **Predictive Maintenance**: Convolutional autoencoders for vibration anomaly detection on wind turbines, fleet-level health monitoring, fault diagnosis from spectral analysis.

4. **Building Energy Optimization**: Hybrid physics-ML building energy models, HVAC control with Deep RL (SAC), occupancy prediction for demand-based control.

5. **Carbon and Emissions Monitoring**: Satellite-based methane detection with hyperspectral analysis, CO2 plume modeling, industrial emissions prediction.

6. **Nuclear and Grid Monitoring**: Core temperature prediction, anomaly detection in cooling systems, power quality disturbance classification.

7. **Grid Cybersecurity**: SCADA intrusion detection, sequential pattern mining for attack detection, adversarial robustness for grid models.

Key technical trends shaping energy AI:

- **Physics-Informed ML**: Embedding physical laws (thermodynamics, power flow equations) into neural network architectures and loss functions
- **Probabilistic Forecasting**: Moving from point forecasts to calibrated prediction intervals for risk-aware grid operations
- **Federated Learning**: Privacy-preserving training across utility customers without centralizing sensitive consumption data
- **Edge AI**: Deploying lightweight models on smart meters, inverters, and substation devices for real-time control
- **Digital Twins**: Comprehensive simulation environments for training RL agents without risking real grid operations

The energy transition demands AI systems that are not just accurate, but safe, interpretable, and trustworthy — capable of operating critical infrastructure that millions of people depend on every day.
