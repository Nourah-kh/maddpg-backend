"""
maddpg_networks.py — MADDPG Neural Networks
===============================================================
Contains:
  1. Actor   -- sees own_obs (13D) and outputs action (4D)
  2. Critic  -- sees concat(all_obs, all_actions) and outputs Q-value
  3. MADDPGAgent -- combines Actor + Critic + Target Networks per drone

Architecture:
  Actor  : MLP  13 -> 256 -> 256 -> 4  (tanh output for action bounds)
  Critic : MLP  (13*N + 4*N) -> 256 -> 256 -> 1

Target Networks:
  - slow copy of Actor and Critic
  - updated via soft update: theta_target = tau*theta + (1-tau)*theta_target
  - used only to compute TD target (stabilizes training)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
# Utility: weight initialization
# ═══════════════════════════════════════════════════════════════

def _init_weights(layer: nn.Linear, std: float = 0.1) -> None:
    """Initialize weights to avoid vanishing/exploding gradients"""
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)


# ═══════════════════════════════════════════════════════════════
# Actor Network
# ═══════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    Policy network -- Decentralized Execution
    Sees own_obs (13D) only and outputs action (4D)

    Input  : own_obs  shape=(batch, obs_dim)    obs_dim=13
    Output : action   shape=(batch, act_dim)    act_dim=4
             in [-1, 1], later scaled by action bounds in the environment
    """

    def __init__(
        self,
        obs_dim:    int = 13,
        act_dim:    int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),        # more stable than BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),                        # output in [-1, 1]
        )

        # initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_weights(layer)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ═══════════════════════════════════════════════════════════════
# Critic Network
# ═══════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    Q-Network -- Centralized Training (CTDE)
    Sees observations + actions for all agents and outputs Q-value

    Input  : concat(all_obs, all_actions)
             shape = (batch, obs_dim*N + act_dim*N)
    Output : Q-value  shape = (batch, 1)

    Example with 4 drones:
        all_obs     = 13*4 = 52
        all_actions =  4*4 = 16
        total input = 68
    """

    def __init__(
        self,
        obs_dim:     int = 13,
        act_dim:     int = 4,
        num_agents:  int = 4,
        hidden_dim:  int = 256,
    ) -> None:
        super().__init__()

        input_dim = obs_dim * num_agents + act_dim * num_agents

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_weights(layer)

    def forward(
        self,
        all_obs:     torch.Tensor,   # (batch, obs_dim * N)
        all_actions: torch.Tensor,   # (batch, act_dim * N)
    ) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# MADDPGAgent -- Actor + Critic + Target Networks per drone
# ═══════════════════════════════════════════════════════════════

class MADDPGAgent(nn.Module):
    """
    Full set of networks for one agent:
      - actor         : policy (for execution)
      - critic        : Q-function (for training only)
      - target_actor  : slow copy of Actor (for TD target)
      - target_critic : slow copy of Critic (for TD target)
      - actor_optimizer
      - critic_optimizer
    """

    def __init__(
        self,
        obs_dim:       int   = 13,
        act_dim:       int   = 4,
        num_agents:    int   = 4,
        hidden_dim:    int   = 256,
        actor_lr:      float = 1e-4,
        critic_lr:     float = 1e-3,
        device:        str   = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)

        # -- Networks ---------------------------------------------------
        self.actor  = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agents, hidden_dim).to(self.device)

        # -- Target Networks (deep copy, frozen until soft update) -------
        self.target_actor  = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

        # freeze targets initially (not trained directly)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # -- Optimizers -------------------------------------------------
        self.actor_optimizer  = torch.optim.Adam(
            self.actor.parameters(),  lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    # ── Soft Update ───────────────────────────────────────────

    def soft_update(self, tau: float = 0.01) -> None:
        """
        theta_target = tau * theta_online + (1 - tau) * theta_target

        Small tau (0.01) = slow update = more stability
        """
        for t_param, o_param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)

        for t_param, o_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)

    # ── Action selection ──────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        obs:         np.ndarray,
        noise_scale: float = 0.0,
        act_bounds:  tuple = (-1.0, 1.0),
    ) -> np.ndarray:
        """
        Returns action with Gaussian noise for exploration.

        noise_scale: zero during evaluation, positive during training
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs_t).squeeze(0).cpu().numpy()

        if noise_scale > 0:
            noise  = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise

        return np.clip(action, act_bounds[0], act_bounds[1]).astype(np.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Used for computing actor loss"""
        return self.actor(obs)


# ═══════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Experience Replay Buffer for MADDPG

    Stores transitions for all agents together:
        (obs_n, actions_n, rewards_n, next_obs_n, dones_n)

    Where n = number of agents

    Off-policy: learn from past experiences -> better sample efficiency
    """

    def __init__(
        self,
        capacity:   int = 100_000,
        num_agents: int = 4,
        obs_dim:    int = 13,
        act_dim:    int = 4,
    ) -> None:

        self.capacity   = capacity
        self.num_agents = num_agents
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.ptr        = 0       # pointer to current position
        self.size       = 0       # number of stored elements

        # Pre-allocate numpy arrays for speed
        self.obs      = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, num_agents, act_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, num_agents),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.dones    = np.zeros((capacity, num_agents),          dtype=np.float32)

    def push(
        self,
        obs:      np.ndarray,   # (num_agents, obs_dim)
        actions:  np.ndarray,   # (num_agents, act_dim)
        rewards:  np.ndarray,   # (num_agents,)
        next_obs: np.ndarray,   # (num_agents, obs_dim)
        dones:    np.ndarray,   # (num_agents,)
    ) -> None:
        """Add a new transition to the buffer"""
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = actions
        self.rewards[self.ptr]  = rewards
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = dones

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> dict:
        """
        Sample a random batch from the buffer

        Returns dict containing tensors ready for GPU/CPU
        """
        assert self.size >= batch_size, (
            f"Buffer has only {self.size} samples, need {batch_size}"
        )

        idx = np.random.randint(0, self.size, size=batch_size)

        def t(x):
            return torch.FloatTensor(x).to(device)

        return {
            "obs":      t(self.obs[idx]),       # (B, N, obs_dim)
            "actions":  t(self.actions[idx]),   # (B, N, act_dim)
            "rewards":  t(self.rewards[idx]),   # (B, N)
            "next_obs": t(self.next_obs[idx]),  # (B, N, obs_dim)
            "dones":    t(self.dones[idx]),     # (B, N)
        }

    def __len__(self) -> int:
        return self.size

    @property
    def is_ready(self) -> bool:
        """Is the buffer ready for training?"""
        return self.size >= 1000   # warmup threshold


# ═══════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("MADDPG Networks — Sanity Check")
    print("=" * 50)

    N          = 4    # number of agents
    OBS_DIM    = 13
    ACT_DIM    = 4
    BATCH      = 32
    HIDDEN     = 256
    DEVICE     = "cpu"

    # ── Test Actor ────────────────────────────────────────────
    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN)
    obs   = torch.randn(BATCH, OBS_DIM)
    act   = actor(obs)
    assert act.shape == (BATCH, ACT_DIM), f"Actor output shape error: {act.shape}"
    assert act.abs().max() <= 1.0 + 1e-5, "Actor output out of [-1,1]"
    print(f"✅ Actor  : input {obs.shape} → output {act.shape}")

    # ── Test Critic ───────────────────────────────────────────
    critic    = Critic(OBS_DIM, ACT_DIM, N, HIDDEN)
    all_obs   = torch.randn(BATCH, OBS_DIM * N)
    all_acts  = torch.randn(BATCH, ACT_DIM * N)
    q_val     = critic(all_obs, all_acts)
    assert q_val.shape == (BATCH, 1), f"Critic output shape error: {q_val.shape}"
    print(f"✅ Critic : input {(all_obs.shape, all_acts.shape)} → Q {q_val.shape}")

    # ── Test MADDPGAgent ──────────────────────────────────────
    agent  = MADDPGAgent(OBS_DIM, ACT_DIM, N, HIDDEN, device=DEVICE)
    obs_np = np.random.randn(OBS_DIM).astype(np.float32)
    action = agent.act(obs_np, noise_scale=0.1)
    assert action.shape == (ACT_DIM,), f"Agent act shape error: {action.shape}"
    print(f"✅ MADDPGAgent.act : obs {obs_np.shape} → action {action.shape}")

    # ── Soft update ───────────────────────────────────────────
    agent.soft_update(tau=0.01)
    print("✅ Soft update : OK")

    # ── Test ReplayBuffer ─────────────────────────────────────
    buf = ReplayBuffer(capacity=10_000, num_agents=N,
                       obs_dim=OBS_DIM, act_dim=ACT_DIM)
    for _ in range(1500):
        buf.push(
            obs      = np.random.randn(N, OBS_DIM).astype(np.float32),
            actions  = np.random.randn(N, ACT_DIM).astype(np.float32),
            rewards  = np.random.randn(N).astype(np.float32),
            next_obs = np.random.randn(N, OBS_DIM).astype(np.float32),
            dones    = np.zeros(N, dtype=np.float32),
        )
    assert buf.is_ready, "Buffer not ready after 1500 pushes"
    batch = buf.sample(BATCH, torch.device(DEVICE))
    assert batch["obs"].shape == (BATCH, N, OBS_DIM)
    print(f"✅ ReplayBuffer : size={len(buf)}, sample obs {batch['obs'].shape}")

    print("=" * 50)
    print("All checks passed ✅")