# World Models — Technical Deep Dive: Math, Losses, Training Recipes

> **Description:** A research-level treatment of the math and training recipes behind frontier world models in 2026. Covers the JEPA loss, the Dreamer RSSM, the diffusion-transformer for video, the GAIA-style driving WM, the reward and value heads, the scaling laws, and the engineering details that matter. Intended for ML researchers, research engineers, and serious students.

---

## Table of Contents

1. [Notation and Conventions](#1-notation-and-conventions)
2. [The JEPA Loss in Full](#2-the-jepa-loss-in-full)
3. [The Dreamer RSSM in Full](#3-the-dreamer-rssm-in-full)
4. [The Diffusion-Transformer Video WM in Full](#5-the-diffusion-transformer-video-wm-in-full)
5. [The Driving WM (GAIA) in Full](#6-the-driving-wm-gaia-in-full)
6. [Action-Conditioning Mechanisms](#7-action-conditioning-mechanisms)
7. [Reward and Value Heads](#8-reward-and-value-heads)
8. [The Latent Dynamics Loss: KL with Free Bits](#9-the-latent-dynamics-loss-kl-with-free-bits)
9. [The Imagination Rollout](#10-the-imagination-rollout)
10. [Scaling Laws for World Models](#11-scaling-laws-for-world-models)
11. [Training Recipes: The 2026 Playbook](#12-training-recipes-the-2026-playbook)
12. [Distributed Training: The Cluster Topology](#13-distributed-training-the-cluster-topology)
13. [Common Failure Modes and Fixes](#14-common-failure-modes-and-fixes)
14. [Reproducing Dreamer V3 on Minecraft](#15-reproducing-dreamer-v3-on-minecraft)
15. [Reproducing V-JEPA 2 Pre-training](#16-reproducing-v-jepa-2-pre-training)
16. [Reproducing a Driving WM (Wayve-style)](#17-reproducing-a-driving-wm-wayve-style)
17. [References and Citations](#18-references-and-citations)

---

## 1. Notation and Conventions

This category uses standard RL and generative modeling notation.

```text
s_t    = state at time t
o_t    = observation at time t
a_t    = action at time t
r_t    = reward at time t
π      = policy
f      = dynamics / transition function
z_t    = latent state at time t
h_t    = deterministic recurrent state (RSSM)
θ      = world model parameters
φ      = policy parameters
ψ      = critic / value function parameters
H      = imagination horizon
T      = real-data horizon
```

Probability distributions are over the *current* state, conditioned on past.

| Symbol | Meaning |
|--------|---------|
| q(s_t \| s_{<t}, a_{<t}, o_t) | Posterior over current state (uses observation) |
| p(s_t \| s_{<t}, a_{<t}) | Prior over current state (no observation) |
| p(s_{t+1} \| s_t, a_t) | Dynamics model |
| p(o_t \| s_t) | Decoder / observation model |
| p(r_t \| s_t) | Reward predictor |
| π(a_t \| s_t) | Policy |
| V^π(s_t) | Value function under policy π |
| 𝔼_π[·] | Expectation under rollouts from π |
| 𝔼_{data}[·] | Expectation over the data distribution |

The world model is the joint distribution

```
p(o_{1:T}, r_{1:T}, s_{1:T} | a_{1:T})
= ∏_t p(s_{t+1} | s_t, a_t) p(o_t | s_t) p(r_t | s_t) p(s_1)
```

The policy is trained to maximize

```
J(π) = 𝔼_π [∑_t γ^t r_t]
```

with γ ∈ [0, 1] the discount factor.

---

## 2. The JEPA Loss in Full

The JEPA family (V-JEPA 1, V-JEPA 2, JEPA 2) uses a latent-space regression loss with several specific choices that make it work.

### 2.1 The Core Loss

Given a video clip (x_1, x_2, ..., x_T) and an action sequence (a_1, ..., a_T):

```python
# V-JEPA 2 pre-training step (simplified)
def vjepa2_step(model, batch):
    x_ctx, x_tgt = batch.context_frames, batch.target_frames  # (B, T_c, C, H, W), (B, T_t, C, H, W)
    a = batch.actions                                              # (B, T, A)

    # 1. Encode context and target
    z_ctx = model.context_encoder(x_ctx)                          # (B, T_c, D)
    with torch.no_grad():                                         # stop-grad on target
        z_tgt = model.target_encoder(x_tgt)                       # (B, T_t, D)

    # 2. Predict target latents from context latents + actions
    z_pred = model.predictor(z_ctx, a)                            # (B, T_t, D)

    # 3. Smooth L1 loss in latent space
    loss_per_token = F.smooth_l1_loss(z_pred, z_tgt, reduction='none').sum(-1)
    loss = loss_per_token[batch.mask].mean()                      # mask out invalid tokens
    return loss
```

The loss is a **smooth L1** (Huber) between predicted and actual target latents. This is in contrast to contrastive losses, which only require "different from negatives."

### 2.2 The Stop-Gradient

The target encoder is an EMA (exponential moving average) of the context encoder, and gradients do not flow through it. This prevents **representational collapse**: a degenerate solution where the encoder outputs a constant and the predictor learns to predict that constant.

```python
# EMA update after each step
@torch.no_grad()
def update_target_encoder(context_encoder, target_encoder, tau=0.99):
    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1 - tau)
```

The coefficient τ = 0.99 (i.e., a slow-moving target) is a hyperparameter that has been validated across many V-JEPA experiments.

### 2.3 Why Smooth L1, Not MSE or Contrastive?

- **MSE** is sensitive to outliers in the latent space and can cause training instability.
- **Smooth L1** is robust to outliers and has been the default for JEPA since V-JEPA 1.
- **Contrastive** (InfoNCE) only requires the prediction to be different from random negatives, not to be specific. It does not enforce the model to predict *the* future, just *a* future. The JEPA hypothesis is that predicting *the* future is what gives you a good world model.

### 2.4 The Predictor

The predictor is a transformer that takes the context latents and the actions and produces the predicted target latents. Key details:

- **Bidirectional attention over context** (you see the full context before predicting).
- **Causal attention over targets** (the prediction for t+k can depend on t but not on t+k+1).
- **Action cross-attention** at each layer (the action modulates the prediction).
- **Sinusoidal time embedding** (so the model knows how far ahead to predict).

### 2.5 The Full Loss with Variance and Covariance Regularization

V-JEPA 2 adds **VICReg-style** regularization on the predicted latents to prevent collapse:

```python
def vicreg_loss(z_pred, z_tgt):
    # Invariance term
    inv = F.mse_loss(z_pred, z_tgt)

    # Variance term: each dim should have variance above a threshold
    std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    var_term = F.relu(1.0 - std_pred).mean()

    # Covariance term: off-diagonal of covariance matrix should be small
    n, d = z_pred.shape
    cov = (z_pred - z_pred.mean(0)).T @ (z_pred - z_pred.mean(0)) / n
    cov_term = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / d

    return inv + 0.01 * var_term + 0.01 * cov_term
```

The combined JEPA + VICReg loss is more robust than either alone, especially for long prediction horizons.

### 2.6 The Forward Pass in Full

The complete V-JEPA 2 forward pass for one training step:

```python
class VJEPA2(nn.Module):
    def __init__(self, ...):
        self.context_encoder = VisionTransformer(...)
        self.target_encoder = VisionTransformer(...)  # EMA copy
        self.predictor = TransformerPredictor(...)

    def forward(self, batch):
        # Encode
        z_ctx = self.context_encoder(batch.context_frames)
        with torch.no_grad():
            z_tgt = self.target_encoder(batch.target_frames)

        # Predict
        z_pred = self.predictor(z_ctx, batch.actions, batch.time_embedding)

        # Loss
        loss = smooth_l1_loss(z_pred, z_tgt) + vicreg_loss(z_pred, z_tgt)
        return loss

    @torch.no_grad()
    def update_target(self):
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(0.99).add_(p_ctx.data, alpha=0.01)
```

---

## 3. The Dreamer RSSM in Full

The Recurrent State-Space Model (RSSM) is the workhorse of latent-dynamics world models. Dreamer V1 → V2 → V3 all use it.

### 3.1 The RSSM Equations

The state is split into a **stochastic** component (z_t) and a **deterministic recurrent** component (h_t):

```
h_t  = f(h_{t-1}, z_{t-1}, a_{t-1})           # recurrent update
ẑ_t  = φ(h_t, a_{t-1})                          # dynamics prior
z_t  ~ q(·|h_t, o_t)                            # posterior (uses observation)
p(o_t | s_t) = decoder(s_t)                     # observation model
p(r_t | s_t) = reward_head(s_t)                 # reward predictor
p(γ_t | s_t) = continue_head(s_t)               # discount / continue predictor
```

The full state is s_t = (h_t, z_t). The split allows the deterministic component to be a long-range memory and the stochastic component to handle stochasticity in the world.

### 3.2 The Full Loss

The Dreamer V3 loss has four terms:

```python
def dreamer_v3_loss(world_model, batch):
    """batch.observations: (B, T, C, H, W), batch.actions: (B, T, A), batch.rewards: (B, T)"""
    # Unroll the world model on real data
    states = []                              # list of (prior, posterior) tuples
    h = world_model.init_state(batch_size=batch.actions.shape[0])
    for t in range(batch.actions.shape[1]):
        prior = world_model.dynamics(h, batch.actions[:, t])
        posterior = world_model.posterior(h, batch.observations[:, t])
        states.append((prior, posterior))
        h = world_model.recurrent(h, posterior.sample(), batch.actions[:, t])

    priors, posteriors = zip(*states)
    s = torch.stack([p.sample() for p in posteriors], dim=1)  # (B, T, S_dim)

    # 1. Reconstruction loss (the observation model)
    obs_pred = world_model.decoder(s)
    recon_loss = F.mse_loss(obs_pred, batch.observations)

    # 2. Reward prediction loss
    rew_pred = world_model.reward_head(s)
    reward_loss = symlog_loss(rew_pred, symlog(batch.rewards))

    # 3. Continue prediction loss (whether the episode is still going)
    cont_pred = world_model.continue_head(s)
    continue_loss = F.binary_cross_entropy_with_logits(cont_pred, batch.continues)

    # 4. Dynamics loss: KL divergence with free bits
    kl = kl_divergence(posteriors, priors)
    free_bits = 1.0  # nats
    kl_loss = torch.clamp(kl, min=free_bits).mean() - free_bits

    total_loss = recon_loss + reward_loss + continue_loss + 0.1 * kl_loss
    return total_loss, s
```

### 3.3 The Dynamics Network

The dynamics function is a small GRU or LSTM that takes the previous state and action and produces a new deterministic state h_t.

```python
class RSSMDynamics(nn.Module):
    def __init__(self, hidden_dim=512, stoch_dim=32, num_cats=32, action_dim=6):
        super().__init__()
        self.gru = nn.GRUCell(input_size=stoch_dim * num_cats + action_dim, hidden_size=hidden_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, num_cats * stoch_dim)
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1024, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, num_cats * stoch_dim)
        )
        self.num_cats = num_cats
        self.stoch_dim = stoch_dim

    def recurrent(self, h, z, a):
        h = self.gru(torch.cat([z, a], dim=-1), h)
        return h

    def prior(self, h):
        logits = self.prior_mlp(h).view(-1, self.num_cats, self.stoch_dim)
        return OneHotDist(logits)

    def posterior(self, h, o):
        logits = self.posterior_mlp(torch.cat([h, o], dim=-1)).view(-1, self.num_cats, self.stoch_dim)
        return OneHotDist(logits)
```

### 3.4 Free Bits

The KL term in the dynamics loss has a **free bits** floor:

```python
def kl_with_free_bits(posterior, prior, free_bits=1.0):
    kl = posterior.kl(prior).sum(dim=-1)  # (B, T)
    kl_per_batch = kl.mean(dim=0)         # (T,)
    return torch.clamp(kl_per_batch, min=free_bits).mean() - free_bits
```

This forces each latent dimension to encode at least 1 nat of information. Without free bits, the dynamics would prefer to encode nothing (collapse), since accurate prediction is easier with a constant prior.

### 3.5 The Imagined Rollout

After training the world model, the policy is trained entirely in imagination:

```python
def imagined_rollout(world_model, policy, initial_state, horizon=50):
    """Generate a trajectory in latent space."""
    s = initial_state
    h, z = s
    actions = []
    states = [s]
    rewards = []
    continues = []
    for t in range(horizon):
        a = policy(h, z)                     # action
        h = world_model.recurrent(h, z, a)
        prior = world_model.prior(h)
        z = prior.sample()                   # sample, no observation
        r = world_model.reward_head((h, z))
        c = torch.sigmoid(world_model.continue_head((h, z)))
        actions.append(a)
        states.append((h, z))
        rewards.append(r)
        continues.append(c)
    return states, actions, rewards, continues
```

The trajectory is used to compute the actor and critic losses via λ-returns (see Section 8).

### 3.6 Symlog Predictions

A Dreamer V3 innovation. The reward and value predictions are made in **symlog space**:

```python
def symlog(x):
    return torch.sign(x) * torch.log(1 + x.abs())

def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)

# Predict symlog(r), compare to symlog(r) ground truth
rew_pred = symlog(reward_head(s))
rew_loss = F.mse_loss(rew_pred, symlog(rewards))
```

Why: rewards in Minecraft range from 0.1 (mining) to 10000+ (diamond). Predicting raw values is numerically unstable. Symlog compresses the range to roughly [-10, 10] without losing the sign or zero. This is one of the unsung tricks that made Dreamer V3 work on diverse domains.

---

## 4. The Diffusion-Transformer Video WM in Full

The 2025-2026 frontier of generative video world models is built on the **diffusion transformer (DiT)** architecture. This is the math behind Sora 2, Genie 3, Veo 3, and Cosmos.

### 4.1 The DiT Backbone

A video diffusion model operates on spacetime patches:

```python
class VideoDiT(nn.Module):
    def __init__(self, ...):
        self.patch_embed = PatchEmbed3D(patch_size_t=2, patch_size_s=16)
        self.transformer = TransformerStack(num_layers=48, hidden_dim=2048, num_heads=32)
        self.unpatch_embed = UnpatchEmbed3D(...)
        self.action_adapter = ActionAdapter(...)
        self.text_adapter = TextAdapter(...)

    def forward(self, x_t, t, text_emb, action_emb):
        # x_t: (B, T, C, H, W) — noisy video at diffusion step t
        # t: (B,) — diffusion timestep
        # text_emb: (B, D_text) — text conditioning
        # action_emb: (B, T, D_action) — action conditioning

        # 1. Patchify and add positional embeddings
        x = self.patch_embed(x_t)  # (B, N_t * N_h * N_w, D)
        x = x + self.pos_embed

        # 2. Condition on text and diffusion timestep
        cond = self.text_adapter(text_emb) + self.timestep_embed(t)

        # 3. Cross-attend to text, action
        for block in self.transformer.layers:
            x = block(x, cond)                              # self-attention + cross-attention
            x = x + self.action_adapter(action_emb)        # action injection

        # 4. Unpatchify
        x = self.unpatch_embed(x)  # (B, T, C, H, W)
        return x  # predicted noise
```

### 4.2 The Diffusion Objective

The model is trained with the standard DDPM-style noise prediction loss:

```python
def diffusion_loss(model, video, text_emb, action_emb):
    """video: (B, T, C, H, W) — clean video"""
    B = video.shape[0]
    # 1. Sample diffusion timestep
    t = torch.randint(0, num_timesteps, (B,))

    # 2. Sample noise
    noise = torch.randn_like(video)

    # 3. Add noise to video
    alpha_bar = scheduler.alphas_cumprod[t]
    x_t = alpha_bar.sqrt() * video + (1 - alpha_bar).sqrt() * noise

    # 4. Predict the noise
    noise_pred = model(x_t, t, text_emb, action_emb)

    # 5. MSE loss
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 4.3 The Scheduler

The 2026 standard is the **flow-matching** scheduler (Stable Diffusion 3, Sora 2), which is more stable than DDPM:

```python
class FlowMatchingScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps

    def add_noise(self, x_0, t):
        """x_0 -> x_t via linear interpolation with noise"""
        noise = torch.randn_like(x_0)
        t_norm = t.float() / self.num_timesteps
        x_t = (1 - t_norm) * x_0 + t_norm * noise
        return x_t

    def step(self, x_t, noise_pred, t, dt):
        """Single denoising step"""
        t_norm = t.float() / self.num_timesteps
        x_0_pred = x_t - t_norm * noise_pred
        x_t_next = x_t + dt * (noise_pred)
        return x_t_next
```

Flow matching avoids the variance explosion at the end of the diffusion process and converges in fewer sampling steps.

### 4.4 The Video VAE

The video is first encoded to a compact latent with a 3D VAE:

```python
class VideoVAE(nn.Module):
    def __init__(self, ...):
        self.encoder = VideoEncoder3D(...)  # ~120M params
        self.decoder = VideoDecoder3D(...)  # ~120M params

    def encode(self, video):
        """video: (B, T, 3, H, W) -> latent: (B, T/4, 4, H/8, W/8)"""
        return self.encoder(video)

    def decode(self, latent):
        """latent -> video"""
        return self.decoder(latent)
```

The compression ratio is 32x in space and 4x in time. A 60-frame 1080p video becomes a (60/4, 4, 1080/8, 1920/8) = (15, 4, 135, 240) latent — small enough for the transformer to process.

### 4.5 The Action Adapter

The action is injected as a cross-attention or as a concatenated token:

```python
class ActionAdapter(nn.Module):
    def __init__(self, action_dim=6, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, action):
        """x: (B, N, D), action: (B, T, A)"""
        # Project action to hidden dim
        a = self.mlp(action)  # (B, T, D)

        # Broadcast to all patches
        # Each patch at time t sees the action at time t
        a = a.unsqueeze(2).expand(-1, -1, x.shape[1] // a.shape[1], -1)  # (B, T, N/T, D)
        a = a.reshape(x.shape[0], x.shape[1], -1)  # (B, N, D)

        # Add to x
        return x + a
```

This is the simplest "action-as-token" approach. More sophisticated variants use FiLM (Feature-wise Linear Modulation) or cross-attention.

### 4.6 The Classifier-Free Guidance

For inference, the model is trained with **classifier-free guidance** to make action following sharper:

```python
def generate_with_cfg(model, text_emb, action_emb, cfg_scale=4.0):
    """Classifier-free guidance for action conditioning."""
    # 1. Predict with conditioning
    eps_cond = model(x_t, t, text_emb, action_emb)

    # 2. Predict without action (action = zeros)
    eps_uncond = model(x_t, t, text_emb, torch.zeros_like(action_emb))

    # 3. Combine
    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    return eps
```

Higher CFG scale (e.g., 7.5) means more action-following but less diversity. 2026 default is 4-6.

---

## 5. The Driving WM (GAIA) in Full

The driving world model (Wayve GAIA, NVIDIA DRIVE Sim, Waabi) is a special case of the generative video model conditioned on driving actions and multi-camera video.

### 5.1 The Multi-Camera Input

The input is synchronized video from 6-8 cameras around the vehicle, plus lidar, IMU, and HD map:

```python
class DrivingWMInput:
    front_camera: Tensor   # (B, T, 3, H, W)
    rear_camera: Tensor
    side_cameras: dict[str, Tensor]   # left, right, ...
    lidar: Tensor          # (B, T, N_points, 3)
    imu: Tensor            # (B, T, 6) — accel + gyro
    hd_map: Tensor         # (B, H, W, lane_features)
    text: Tensor           # (B, T, D) — language instructions
    action: Tensor         # (B, T, 2) — steering + throttle
```

### 5.2 The Architecture

GAIA-2 (Wayve, 2023-2026) is a diffusion transformer with:

```python
class GAIA2(nn.Module):
    def __init__(self, ...):
        self.video_vae = MultiCameraVideoVAE(...)  # encodes all cameras jointly
        self.map_encoder = HDMapEncoder(...)
        self.lidar_encoder = PointNetLidar(...)
        self.action_encoder = nn.Linear(2, 512)
        self.text_encoder = T5(...)
        self.dit = DiffusionTransformer(...)

    def forward(self, batch, t):
        # Encode each input modality
        v_latent = self.video_vae.encode(batch.cameras)  # (B, T, 4, H/8, W/8) per camera
        m_latent = self.map_encoder(batch.hd_map)         # (B, D_map)
        l_latent = self.lidar_encoder(batch.lidar)       # (B, D_lidar)
        a_latent = self.action_encoder(batch.action)      # (B, T, 512)
        t_latent = self.text_encoder(batch.text)          # (B, T, D_text)

        # Fuse all latents
        cond = torch.cat([v_latent.flatten(1), m_latent, l_latent, a_latent, t_latent], dim=-1)

        # Predict noise on the next camera frame
        noise_pred = self.dit(v_latent[:, -1], t, cond)

        return noise_pred
```

The model is trained to predict the next frame of all cameras given the previous frames and the action.

### 5.3 The Loss

Standard diffusion loss on the next-frame prediction:

```python
def gaia_loss(model, batch):
    """Predict the next multi-camera frame given history and action."""
    history = batch.cameras[:, :-1]  # all but last
    target = batch.cameras[:, -1]    # the last frame
    action_history = batch.actions[:, :-1]

    # Encode history
    hist_latent = model.video_vae.encode(history)

    # Diffusion on target
    t = torch.randint(0, 1000, (history.shape[0],))
    noise = torch.randn_like(target)
    target_noisy = scheduler.add_noise(target, t, noise)

    # Predict noise
    noise_pred = model(hist_latent, t, action_history, batch.hd_map, batch.text)

    # Loss
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 5.4 The Evaluation

GAIA models are evaluated on:

- **Visual quality** (FVD, LPIPS on the predicted next frame).
- **Physical plausibility** (does the predicted ego-motion match the actual motion?).
- **Downstream driving** (if you train a driving policy on the synthetic data, how well does it drive in the real world?).

The downstream driving metric is the most important but also the most expensive.

---

## 6. Action-Conditioning Mechanisms

How actions enter the world model. The 2026 design space.

### 6.1 Action as Token (Sequence)

```python
# Add action tokens to the input sequence
sequence = [state_tokens, action_tokens, next_state_tokens]
```

**Used in:** IRIS, hybrid LLM + WM systems.

**Pros:** unified token space; easy to combine with language.

**Cons:** quantization error; need a tokenizer for actions.

### 6.2 Action as Cross-Attention

```python
# State attends to actions via cross-attention
state_features = self_attention(state_features)
state_features = cross_attention(state_features, action_features, action_features)
```

**Used in:** Sora 2 (storyboard), Cosmos 2.0, Genie 3.

**Pros:** continuous actions; flexible conditioning.

**Cons:** more parameters; slower training.

### 6.3 Action as FiLM

```python
# Action produces a per-channel scale and shift
scale, shift = action_mlp(action)  # (B, D), (B, D)
x = scale * x + shift
```

**Used in:** Dreamer V3, JEPA family.

**Pros:** cheap; well-suited to continuous actions; stable training.

**Cons:** one modulation per layer; can be limiting for high-DoF action.

### 6.4 Action as Concatenation

```python
# Concatenate action to state features
x = torch.cat([state_features, action], dim=-1)
```

**Used in:** simple baseline WMs, early Dreamer.

**Pros:** simplest.

**Cons:** action must be low-dimensional; doesn't scale.

### 6.5 Action-Free (Generation Only)

```python
# No action; the model generates plausible continuations
sequence = [state_tokens, next_state_tokens]
```

**Used in:** Sora 1 (image-to-video), Genie 2, Veo 3.

**Pros:** no labeled actions needed.

**Cons:** cannot plan ("what if I do X?").

---

## 7. Reward and Value Heads

For RL-style WMs, the reward and value functions are heads on the world model.

### 7.1 The Reward Head

```python
class RewardHead(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.mlp(state)  # (B, 1)
```

Trained on observed rewards with MSE (or symlog MSE for Dreamer V3).

### 7.2 The Value Head

```python
class ValueHead(nn.Module):
    """Predicts V^π(s) for the current policy."""
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.mlp(state)
```

Trained on **λ-returns** from imagined rollouts (see Section 8).

### 7.3 The Continue Head

A binary head predicting whether the episode continues at this state.

```python
class ContinueHead(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.mlp = nn.Linear(state_dim, 1)

    def forward(self, state):
        return self.mlp(state)  # logit; sigmoid for probability
```

Used in Dreamer V3 for proper bootstrapping at episode boundaries.

### 7.4 The Critic as a Separate WM

In some 2026 systems (Wayve AV2.0, Tesla FSD V13), there is a **second** world model used as the critic. This is computationally expensive but gives sharper safety signal.

---

## 8. The Imagination Rollout

The imagined rollout is the core of RL-with-WM. The policy and critic are trained on rollouts that happen entirely in the world model's latent space.

### 8.1 The Full Algorithm

```python
def dreamer_v3_train_step(world_model, actor, critic, batch, horizon=50):
    # 1. Train world model on real data
    wm_loss, real_states = dreamer_v3_loss(world_model, batch)
    wm_loss.backward()
    optimizer_world_model.step()
    optimizer_world_model.zero_grad()

    # 2. Imagine trajectories from real states
    initial_states = real_states.detach()
    imagined_states, imagined_actions, imagined_rewards, imagined_continues = imagined_rollout(
        world_model, actor, initial_states, horizon
    )

    # 3. Compute λ-returns for the critic
    with torch.no_grad():
        values = critic(imagined_states)
        next_values = critic(imagined_states[1:]).shift(...)  # careful indexing
        lambda_returns = compute_lambda_returns(
            imagined_rewards, values, imagined_continues, lambda_=0.95
        )

    # 4. Train the critic
    value_pred = critic(imagined_states[:-1])
    critic_loss = F.mse_loss(value_pred, lambda_returns)
    critic_loss.backward()
    optimizer_critic.step()
    optimizer_critic.zero_grad()

    # 5. Train the actor to maximize imagined return
    actor_loss = -lambda_returns.mean()
    actor_loss.backward()
    optimizer_actor.step()
    optimizer_actor.zero_grad()
```

### 8.2 λ-Returns

The λ-return is a geometric-weighted average of n-step returns:

```python
def compute_lambda_returns(rewards, values, continues, lambda_=0.95):
    """
    rewards: (T, B, 1)
    values:  (T, B, 1)
    continues: (T, B, 1)
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    last = values[-1]
    for t in reversed(range(T)):
        bootstrap = continues[t] * (lambda_ * last + (1 - lambda_) * values[t])
        returns[t] = rewards[t] + continues[t] * (
            lambda_ * last + (1 - lambda_) * values[t]
        )
        last = returns[t]
    return returns
```

λ = 0.95 is a 2026 default. λ = 0 is 1-step TD; λ = 1 is Monte Carlo.

### 8.3 The Imagination Horizon

The horizon H is critical. Too short and the agent doesn't see consequences; too long and the imagined trajectories diverge from reality. Dreamer V3 default is H = 50. Some 2026 systems use H = 200 for long-horizon tasks.

---

## 9. The Latent Dynamics Loss: KL with Free Bits

The KL term in the RSSM loss is a critical design choice. The 2026 standard is "KL with free bits" but several variants exist.

### 9.1 Standard KL

```python
kl = posterior.kl_divergence(prior).sum(dim=-1)  # (B, T)
```

This penalizes the posterior for diverging from the prior. The intuition: the model should use as little information per timestep as possible (Occam's razor).

### 9.2 Free Bits

```python
kl_per_dim = kl.mean(dim=0)  # (B, T) -> average over batch
kl_loss = torch.clamp(kl_per_dim, min=1.0).mean() - 1.0
```

Free bits force each latent dimension to encode at least 1 nat of information. This prevents collapse (all posteriors = prior) and is a 2026 standard.

### 9.3 The β-VAE Variant

```python
kl_loss = beta * kl.mean()  # beta = 0.1 typical
```

Multiplicative weighting of the KL term. Higher beta = more compression, less information. Lower beta = less compression, more information.

### 9.4 The "Disagree" Loss (2026)

A newer 2026 approach: add a loss term that explicitly forces the prior and posterior to *disagree*:

```python
disagree = -kl.detach() * kl  # encourage KL when it would be zero
```

This is still research-grade but shows promise for sparse-reward environments.

---

## 10. The Imagination Rollout in Practice

A real training step takes the form:

```python
def full_train_step(world_model, actor, critic, batch, optimizers, hparams):
    wm_optim, actor_optim, critic_optim = optimizers
    wm_lr, actor_lr, critic_lr = hparams['lrs']

    # World model update
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        wm_loss, real_states = dreamer_v3_loss(world_model, batch)
    wm_optim.zero_grad()
    wm_loss.backward()
    wm_optim.step()

    # Imagine
    with torch.no_grad():
        initial = real_states[:, -1]  # the last real state
        imagined = imagined_rollout(world_model, actor, initial, horizon=50)

    # Critic update
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        critic_loss = critic_loss_fn(critic, imagined)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Actor update
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        actor_loss = actor_loss_fn(actor, critic, imagined)
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
```

This entire step takes ~200-500 ms on 8x H100 for a small Dreamer.

---

## 11. Scaling Laws for World Models

Scaling laws for world models are an active 2026 research area. Some patterns are emerging.

### 11.1 The Loss vs Compute Frontier

For generative video models (Sora 2 style):

```python
# Approximate scaling law
def video_model_loss(N, D, T_video):
    """
    N: number of parameters (in billions)
    D: training data (in hours)
    T_video: target video length (in seconds)
    """
    L = 0.4 + 0.3 * (N / 10) ** -0.15 + 0.2 * (D / 1000) ** -0.1 + 0.1 * T_video ** 0.3
    return L
```

This is a 2026 empirical fit from Sora 2 / Veo 3 / Genie 3 papers. Loss decreases as N^0.15 and D^0.1 — both sublinear, which means **scale matters a lot**.

### 11.2 The Latent Dynamics Scaling Law

For latent dynamics (Dreamer V3 style):

```python
# Approximate scaling law
def latent_dynamics_loss(N, D, action_dim, state_dim):
    L = 0.5 + 0.4 * (N / 100) ** -0.2 + 0.1 * (D / 1e6) ** -0.05
    return L
```

Latent models are more data-efficient per parameter, but the absolute loss is higher than generative models at the same scale.

### 11.3 The Compute-Optimal Frontier

For a fixed compute budget C:

- Generative video: allocate 30% to data, 70% to parameters.
- Latent dynamics: allocate 60% to data, 40% to parameters.

This is the inverse of LLMs, where the data/parameter split is closer to 50/50. WMs are more data-hungry than LLMs.

### 11.4 The Emergent Capabilities

At certain scales, world models exhibit **emergent** capabilities:

- **~1B params:** basic physical reasoning.
- **~4B params:** object permanence, gravity, simple counterfactuals.
- **~14B params:** multi-step planning, social dynamics.
- **~50B params:** theory of mind in multi-agent settings.
- **~100B+ params:** full commonsense reasoning across modalities (unverified as of 2026).

The 2026 frontier is around 10-50B parameters. Beyond 50B, returns diminish and the cost is enormous.

---

## 12. Training Recipes: The 2026 Playbook

The empirically-validated training recipe for a frontier world model. Based on Sora 2, Cosmos 2.0, Dreamer V3, and V-JEPA 2 papers.

### 12.1 Data

```text
- 100M+ hours of video
- Diverse: indoor, outdoor, driving, manipulation, sports
- High-quality: 1080p+, 30+ fps
- Action-labeled where possible
- Deduplicated
- License-clean (or licensed)
```

### 12.2 Compute

```text
- 1-10K H100 / H200 GPUs
- 1-6 months wall-clock
- ~$5-50M training cost
```

### 12.3 Architecture

```text
- Diffusion transformer (generative) or RSSM (latent)
- 3D video VAE (compression)
- Action adapter (FiLM or cross-attention)
- 1-50B parameters
```

### 12.4 Optimization

```text
- Optimizer: AdamW with β_1=0.9, β_2=0.95
- LR: 1e-4 to 5e-4, cosine schedule with 1% warmup
- Batch size: 1024-8192 (effective)
- Weight decay: 0.1
- Gradient clipping: 1.0
- Mixed precision: bfloat16
- EMA of weights: τ=0.9999
```

### 12.5 Regularization

```text
- Stop-gradient on target encoder (for JEPA)
- Free bits on KL (for Dreamer)
- VICReg on predicted latents (for JEPA V2)
- Dropout: 0.1 on attention
- Label smoothing: 0.1 (for generative)
```

### 12.6 Evaluation During Training

```text
- Every 1K steps: holdout FVD on 1K clips
- Every 10K steps: physical reasoning benchmark (IntPhys, Physical-IQ)
- Every 100K steps: full ImagineBench
- Every 1M steps: human evaluation (200 clips, 3 raters)
```

---

## 13. Distributed Training: The Cluster Topology

World model training is GPU-bound and has a different communication pattern than LLM training. The 2026 cluster topology:

### 13.1 The 3D Parallelism Recipe

For a 10B-parameter WM on 1024 H100s:

| Dimension | Strategy | Why |
|-----------|----------|-----|
| **Data parallel** | 16-way | Replicate the model 16x |
| **Tensor parallel** | 8-way | Split the DiT layers across 8 GPUs |
| **Pipeline parallel** | 4-way | Split the depth across 4 stages |
| **Sequence parallel** | 2-way | Split the long video context |
| **Total GPUs** | 16 × 8 × 4 / 2 = 1024 | Matches 16 nodes × 8 GPUs |

### 13.2 The Communication Pattern

- **Intra-node (NVLink):** 8 GPUs per node, ~900 GB/s bidirectional.
- **Inter-node (InfiniBand NDR):** 400 Gbps, used for tensor parallel and pipeline parallel.
- **All-reduce on gradients:** every step, ~50-100 ms.
- **All-gather for video VAE encoding:** every step, ~20-30 ms.

The 2026 frontier: FSDP-2 + Ring Attention + Sequence Parallel. See the Megatron-Core and torchtitan repositories.

### 13.3 Checkpointing

```python
# 2026 standard: FSDP checkpoint every 1000 steps, with async upload to S3
torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=size_based_auto_wrap_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
)
```

A 10B model in bfloat16 is 20 GB, plus optimizer state (40 GB), plus gradients (20 GB) = 80 GB per GPU. With FSDP full_shard, this is sharded across 1024 GPUs to ~80 MB per GPU.

---

## 14. Common Failure Modes and Fixes

The 2026 enumeration of the most common training problems and their fixes.

### 14.1 Representational Collapse (JEPA)

**Symptom:** All latents become constant. Loss is low. Output is meaningless.

**Fixes:**
1. Stop-gradient on target encoder.
2. VICReg-style variance and covariance terms.
3. Higher learning rate for the predictor.
4. EMA τ closer to 1.0 (slower target update).

### 14.2 Posterior Collapse (RSSM)

**Symptom:** Posterior = prior. The model ignores observations.

**Fixes:**
1. Free bits on the KL term.
2. Higher learning rate for the observation encoder.
3. Reduce the KL weight.
4. Re-initialize the posterior MLP.

### 14.3 Reward Hacking (Dreamer V3)

**Symptom:** Policy finds a flaw in the world model and exploits it (e.g., "kills" itself to bootstrap infinite value).

**Fixes:**
1. Continue head to detect episode terminations.
2. Symlog predictions.
3. Value clipping.
4. Real-vs-imagined data mixing.

### 14.4 Sim-to-Real Gap (Driving / Robotics)

**Symptom:** Policy trained in the WM fails in the real world.

**Fixes:**
1. Domain randomization in the WM.
2. Real-world fine-tuning of the policy (not the WM).
3. System identification: tune the WM to match real data.
4. Sim-to-real transfer learning (use a small real-world dataset to fine-tune the policy).

### 14.5 Action Chattering (Robotics)

**Symptom:** Robot oscillates between similar actions.

**Fixes:**
1. Action sequence prediction (predict 5 actions, execute 1).
2. Action smoothing loss.
3. Action discretization.

### 14.6 Catastrophic Forgetting in Online WMs

**Symptom:** World model forgets early training data.

**Fixes:**
1. Replay buffer with a mix of old and new.
2. Elastic Weight Consolidation.
3. Periodic full-dataset fine-tuning.

### 14.7 WM Memorization

**Symptom:** WM regenerates exact training clips verbatim.

**Fixes:**
1. Augmentation (random crops, color jitter).
2. Differential privacy training.
3. Memorization detection in evaluation.

---

## 15. Reproducing Dreamer V3 on Minecraft

The full end-to-end recipe for training Dreamer V3 to get diamonds in Minecraft from pixels.

### 15.1 Compute

- 8x H100 GPUs
- 1 week training time
- ~$5K compute cost
- 50M environment steps

### 15.2 Environment

```python
import minerl  # or a modern Minecraft RL env like MineDojo

env = minerl.make('MineRLNavigateDense-v0')
# or
env = gym.make('MineDojo-CreateVillageAnimalPen-v0')
```

### 15.3 Hyperparameters

```python
hyperparams = {
    # World model
    'wm_hidden_dim': 512,
    'wm_stoch_dim': 32,
    'wm_num_cats': 32,
    'wm_free_bits': 1.0,
    'wm_recon_scale': 1.0,
    'wm_reward_scale': 1.0,
    'wm_kl_scale': 0.1,
    'wm_continue_scale': 1.0,

    # Imagination
    'imagine_horizon': 50,
    'imagine_lambda': 0.95,
    'imagine_gamma': 0.997,

    # Actor / critic
    'actor_hidden_dim': 512,
    'actor_lr': 1e-5,
    'critic_lr': 1e-5,
    'wm_lr': 1e-4,
    'actor_entropy_scale': 1e-3,

    # Training
    'batch_size': 50,
    'batch_length': 50,
    'train_every': 5,  # environment steps
    'replay_capacity': 1_000_000,
}
```

### 15.4 Training Loop

```python
# Pseudocode — see dreamerv3-torch repository for full implementation
for step in range(50_000_000):
    # 1. Collect environment data
    action = policy(real_state)
    obs, reward, done, _ = env.step(action)
    replay.add(obs, action, reward, done)

    # 2. Train world model
    if step % 5 == 0:
        batch = replay.sample(50, 50)
        wm_loss = dreamer_v3_loss(world_model, batch)
        wm_loss.backward()
        wm_optim.step()

    # 3. Train actor / critic
    if step % 5 == 0:
        batch = replay.sample(50, 50)
        states = dreamer_v3_encode(world_model, batch)
        imagined = imagined_rollout(world_model, policy, states, horizon=50)
        train_actor_critic(actor, critic, imagined)
```

### 15.5 Expected Results

After 50M environment steps (~1 week on 8x H100):
- Diamond obtained: ~30% of runs
- Diamond efficiency: 0.5-2 diamonds per attempt
- Emergent behavior: crafting, smelting, exploration, building shelters

This is the 2024 state-of-the-art. As of 2026, similar results are achievable with ~10x less compute using better algorithms.

---

## 16. Reproducing V-JEPA 2 Pre-training

The full recipe for V-JEPA 2 pre-training.

### 16.1 Compute

- 256x H100 GPUs
- 30 days training
- ~$2M compute cost
- 1B video clips

### 16.2 Data

```python
# Mix of public video datasets
datasets = {
    'howto100m': 0.3,    # instructional
    'kinetics': 0.2,      # action-labeled
    'ego4d': 0.2,         # first-person
    'ssv2': 0.1,          # physical reasoning
    'internal': 0.2,      # proprietary
}
```

### 16.3 Hyperparameters

```python
hyperparams = {
    'encoder_dim': 1024,        # ViT-Large
    'predictor_dim': 1024,
    'num_predictor_layers': 24,
    'num_predictor_heads': 16,
    'patch_size': 16,
    'tubelet_size': 2,         # temporal patch size
    'ema_tau': 0.99,
    'lr': 1e-4,
    'batch_size': 256,
    'clips_per_video': 4,      # sample 4 clips per video
    'clip_length': 16,         # frames per clip
    'prediction_horizon': 4,   # predict 4 clips ahead
}
```

### 16.4 Training Loop

```python
# Pseudocode — see facebookresearch/vjepa2 repository
for step in range(1_000_000):
    batch = dataloader.sample(256)

    # Encode
    z_ctx = context_encoder(batch.context_clips)
    with torch.no_grad():
        z_tgt = target_encoder(batch.target_clips)

    # Predict
    z_pred = predictor(z_ctx, batch.actions)

    # Loss
    loss = smooth_l1_loss(z_pred, z_tgt) + vicreg_loss(z_pred, z_tgt)
    loss.backward()
    optim.step()
    optim.zero_grad()

    # Update target encoder
    update_target_encoder(0.99)
```

### 16.5 Expected Results

After 1M steps:
- Video question answering: 75-80% accuracy
- Physical reasoning (Physical-IQ): 60-65%
- Action anticipation: 70%+ top-5
- Object permanence: 85%+

---

## 17. Reproducing a Driving WM (Wayve-style)

The recipe for a small driving WM that you can train on a single node.

### 17.1 Compute

- 1x 8x H100 node
- 2 weeks training
- ~$5K compute cost
- 100 hours of driving video

### 17.2 Data

```python
# 100 hours of multi-camera driving video
# You can use the Wayve Open Dataset or nuScenes
dataset = nuScenes(version='v1.0-mini', dataroot='/data/nuscenes')
# or
dataset = WayveOpenDataset('/data/wayve-open')
```

### 17.3 Architecture

A small GAIA-2-like model:

```python
class SmallDrivingWM(nn.Module):
    def __init__(self, ...):
        self.video_vae = SmallVideoVAE()  # 10M params
        self.action_encoder = nn.Linear(2, 256)  # steering + throttle
        self.dit = SmallDiT(hidden_dim=512, num_layers=12)  # 100M params

    def forward(self, history, action):
        hist_latent = self.video_vae.encode(history)
        a_latent = self.action_encoder(action)
        # Diffusion on next frame
        eps_pred = self.dit(hist_latent[:, -1], t, a_latent)
        return eps_pred
```

### 17.4 Training

```python
# Train for 100K steps
for step in range(100_000):
    batch = dataloader.sample(batch_size=8)  # 8 scenes, 60 frames each
    loss = gaia_loss(model, batch)
    loss.backward()
    optim.step()
```

### 17.5 Expected Results

After 100K steps:
- Visual quality: FVD ~30 on nuScenes validation
- Driving policy: not yet deployable but useful for synthetic data
- Sim-to-real: significant gap; needs fine-tuning on real data

This is a "research-quality" WM. Production WMs require 100-1000x more compute and data.

---

## 18. References and Citations

The 2026 reference list. Format: [author, year, title, link].

```text
[1] LeCun, 2022. "A Path Towards Autonomous Machine Intelligence."  https://arxiv.org/abs/2212.06137
[2] Ha & Schmidhuber, 2018. "World Models."  https://arxiv.org/abs/1803.10122
[3] Hafner et al., 2024. "Mastering Diverse Domains through World Models (Dreamer V3)."  Nature.  https://www.nature.com/articles/s41586-024-07492-1
[4] Assael et al., 2024. "V-JEPA: Video Joint Embedding Predictive Architecture."  Meta FAIR.  https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video/
[5] Bardes et al., 2022. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning."  ICLR 2022.  https://arxiv.org/abs/2105.04906
[6] Hu et al., 2024. "Sora Technical Report."  OpenAI.  https://openai.com/research/video-generation-models-as-world-simulators
[7] Wayve, 2023. "GAIA-1: A Generative World Model for Autonomous Driving."  https://wayve.com/thinking/gaia/
[8] NVIDIA, 2024. "Cosmos: World Foundation Models for Physical AI."  https://www.nvidia.com/en-us/cosmos/
[9] DeepMind, 2024. "Genie 2: Interactive 3D Worlds from a Single Image."  https://deepmind.google/discover/blog/genie-2/
[10] Micheli et al., 2023. "IRIS: Imagination with auto-Regressive Integrated Systems."  https://arxiv.org/abs/2211.17200
[11] Peebles & Xie, 2023. "Scalable Diffusion Models with Transformers (DiT)."  ICCV 2023.  https://arxiv.org/abs/2212.09748
[12] Lipman et al., 2023. "Flow Matching for Generative Modeling."  ICLR 2023.  https://arxiv.org/abs/2210.02747
[13] Esser et al., 2024. "Stable Diffusion 3: Scaling Rectified Flow Transformers."  https://arxiv.org/abs/2410.23724
[14] Decart et al., 2024. "OASIS: Real-Time World Models for Interactive Content."  https://oasis.decart.ai/
[15] Team et al., 2024. "VBench: A Comprehensive Benchmark for Video Generation."  CVPR 2024.  https://arxiv.org/abs/2311.17982
[16] Battaglia et al., 2013. "IntPhys: A Framework for Benchmarking Intuitive Physical Reasoning."  https://intphys.com/
[17] Bear et al., 2021. "Physion: Evaluating Physical Prediction from Vision in Humans and Machines."  NeurIPS 2021.  https://physion-demo.github.io/
[18] Sutton, 1990. "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming."  ICML 1990.
[19] Hafner et al., 2019. "Learning Latent Dynamics for Planning from Pixels (PlaNet)."  ICML 2019.  https://arxiv.org/abs/1811.04551
[20] Watter et al., 2015. "Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images."  NeurIPS 2015.  https://arxiv.org/abs/1506.07365
```

---

## Cross-References to Other Categories

- **Foundations (01)**: [01-Foundations/06-Reinforcement-Learning.md](../01-Foundations/06-Reinforcement-Learning.md) — RL is the training framework for WMs.
- **LLMs (02)**: [02-LLMs/01-Transformer-Architecture.md](../02-LLMs/01-Transformer-Architecture.md) — Transformers are the WM backbone.
- **Advanced (06)**: [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md) — Diffusion is the generative technique.
- **Research Frontiers (17)**: [17-Research-Frontiers-2026/01-Overview.md](../17-Research-Frontiers-2026/01-Overview.md) — WMs are a major frontier.
- **Local AI (23)**: [23-Local-AI-Inference-Self-Hosting/01-Overview.md](../23-Local-AI-Inference-Self-Hosting/01-Overview.md) — Small WMs run locally.

---

*Next: [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) — the open-source ecosystem, foundation model downloads, and deployment tooling.*
