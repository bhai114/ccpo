# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Historical Reward Normalizer for multi-agent RL with EMA statistics.
Maintains exponential moving average statistics for reward normalization.
"""

from collections import deque
import numpy as np
import torch


class HistoricalRewardNormalizer:
    """
    Maintains historical reward statistics using Exponential Moving Average (EMA).
    
    This normalizer keeps track of:
    - Agent1 marginal contributions: Δ = R_joint - R_solo
    - Agent2 joint rewards: R_joint
    - Agent2 solo rewards: R_solo
    
    And provides:
    - History-normalized contribution gains for Agent1
    - History-gated joint-solo optimization for Agent2
    """
    
    def __init__(self, buffer_size=1000, epsilon=1e-6, min_samples=10, ema_decay=0.99, 
                 alpha=1.0, eta=1.0):
        """
        Args:
            buffer_size: Maximum number of historical samples to keep (for initial statistics)
            epsilon: Small constant to prevent division by zero
            min_samples: Minimum number of samples before normalization is applied
            ema_decay: EMA decay factor (default: 0.99)
            alpha: Sensitivity parameter for tanh shaping in Agent1 (default: 1.0)
            eta: Gate sharpness parameter for Agent2 (default: 1.0)
        """
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.eta = eta
        
        # Historical buffers for initial statistics (before EMA kicks in)
        self.agent1_deltas_buffer = deque(maxlen=buffer_size)
        self.agent2_joint_buffer = deque(maxlen=buffer_size)
        self.agent2_solo_buffer = deque(maxlen=buffer_size)
        
        # EMA statistics for Agent1 (delta = R_joint - R_solo)
        self.mu_delta = 0.0
        self.sigma_delta = 1.0
        self.m2_delta = 1.0  # Second moment for variance calculation
        
        # EMA statistics for Agent2 joint rewards
        self.mu_joint = 0.0
        self.sigma_joint = 1.0
        self.m2_joint = 1.0
        
        # EMA statistics for Agent2 solo rewards
        self.mu_solo = 0.0
        self.sigma_solo = 1.0
        self.m2_solo = 1.0
        
        # Update counter
        self.n_updates = 0
        self._stats_dirty = True
    
    def update(self, agent1_deltas, agent2_joint_rewards, agent2_solo_rewards):
        """
        Update historical statistics with EMA.
        
        Args:
            agent1_deltas: List or array of marginal contributions (Δ = R_joint - R_solo)
            agent2_joint_rewards: List or array of joint rewards
            agent2_solo_rewards: List or array of solo rewards
        """
        # Convert to numpy arrays
        if isinstance(agent1_deltas, (torch.Tensor, list)):
            agent1_deltas = np.array(agent1_deltas, dtype=np.float32).flatten()
        if isinstance(agent2_joint_rewards, (torch.Tensor, list)):
            agent2_joint_rewards = np.array(agent2_joint_rewards, dtype=np.float32).flatten()
        if isinstance(agent2_solo_rewards, (torch.Tensor, list)):
            agent2_solo_rewards = np.array(agent2_solo_rewards, dtype=np.float32).flatten()
        
        # Update buffers (for initial statistics before EMA kicks in)
        self.agent1_deltas_buffer.extend(agent1_deltas.tolist())
        self.agent2_joint_buffer.extend(agent2_joint_rewards.tolist())
        self.agent2_solo_buffer.extend(agent2_solo_rewards.tolist())
        
        # Compute batch statistics
        batch_delta_mean = float(np.mean(agent1_deltas))
        batch_delta_var = float(np.var(agent1_deltas))
        
        batch_joint_mean = float(np.mean(agent2_joint_rewards))
        batch_joint_var = float(np.var(agent2_joint_rewards))
        
        batch_solo_mean = float(np.mean(agent2_solo_rewards))
        batch_solo_var = float(np.var(agent2_solo_rewards))
        
        # Initialize or update EMA
        if self.n_updates == 0:
            # First update: use batch statistics
            self.mu_delta = batch_delta_mean
            self.m2_delta = batch_delta_var
            self.sigma_delta = np.sqrt(batch_delta_var) if batch_delta_var > self.epsilon else 1.0
            
            self.mu_joint = batch_joint_mean
            self.m2_joint = batch_joint_var
            self.sigma_joint = np.sqrt(batch_joint_var) if batch_joint_var > self.epsilon else 1.0
            
            self.mu_solo = batch_solo_mean
            self.m2_solo = batch_solo_var
            self.sigma_solo = np.sqrt(batch_solo_var) if batch_solo_var > self.epsilon else 1.0
        else:
            # EMA update for delta
            self.mu_delta = self.ema_decay * self.mu_delta + (1 - self.ema_decay) * batch_delta_mean
            self.m2_delta = self.ema_decay * self.m2_delta + (1 - self.ema_decay) * batch_delta_var
            self.sigma_delta = np.sqrt(self.m2_delta) if self.m2_delta > self.epsilon else 1.0
            
            # EMA update for joint
            self.mu_joint = self.ema_decay * self.mu_joint + (1 - self.ema_decay) * batch_joint_mean
            self.m2_joint = self.ema_decay * self.m2_joint + (1 - self.ema_decay) * batch_joint_var
            self.sigma_joint = np.sqrt(self.m2_joint) if self.m2_joint > self.epsilon else 1.0
            
            # EMA update for solo
            self.mu_solo = self.ema_decay * self.mu_solo + (1 - self.ema_decay) * batch_solo_mean
            self.m2_solo = self.ema_decay * self.m2_solo + (1 - self.ema_decay) * batch_solo_var
            self.sigma_solo = np.sqrt(self.m2_solo) if self.m2_solo > self.epsilon else 1.0
        
        self.n_updates += 1
        self._stats_dirty = True
    
    def _update_statistics(self):
        """Statistics are already maintained via EMA, just clear dirty flag."""
        self._stats_dirty = False
    
    def normalize_agent1_rewards(self, agent1_deltas):
        """
        Agent1: History-Aware Contribution Shaping.
        
        Computes: r1 = tanh(α * z^Δ)
        where z^Δ = (Δ - μ_Δ) / (σ_Δ + ε)
        
        Args:
            agent1_deltas: Array or tensor of current marginal contributions (Δ)
                          Shape: (batch_size,) or (batch_size, max_turns)
        
        Returns:
            Shaped rewards with same shape as input
        """
        self._update_statistics()
        
        # Check if we have enough historical data
        if self.n_updates < 1 or len(self.agent1_deltas_buffer) < self.min_samples:
            print(f"[HistoricalNormalizer] Not enough agent1 history (updates={self.n_updates}, buffer={len(self.agent1_deltas_buffer)}), skipping normalization")
            return agent1_deltas
        
        # Convert to numpy if needed
        is_tensor = isinstance(agent1_deltas, torch.Tensor)
        if is_tensor:
            device = agent1_deltas.device
            dtype = agent1_deltas.dtype
            agent1_deltas = agent1_deltas.cpu().numpy()
        
        # Step 1: History-normalize contribution gain
        z_delta = (agent1_deltas - self.mu_delta) / (self.sigma_delta + self.epsilon)
        
        # Step 2: Apply non-linear tanh shaping
        r1 = np.tanh(self.alpha * z_delta)
        
        if is_tensor:
            r1 = torch.tensor(r1, dtype=dtype, device=device)
        
        return r1
    
    def normalize_agent2_rewards(self, agent2_joint_rewards, agent2_solo_rewards):
        """
        Agent2: History-Gated Joint-Solo Optimization.
        
        Computes: r2 = g * z^joint + (1-g) * z^solo
        where:
            z^joint = (R_joint - μ_joint) / (σ_joint + ε)
            z^solo = (R_solo - μ_solo) / (σ_solo + ε)
            g = sigmoid(η * μ_Δ / (σ_Δ + ε))
        
        Args:
            agent2_joint_rewards: Array or tensor of current joint rewards
                                 Shape: (batch_size,)
            agent2_solo_rewards: Array or tensor of current solo rewards
                                Shape: (batch_size,)
        
        Returns:
            Gated rewards with shape (batch_size,)
        """
        self._update_statistics()
        
        # Check if we have enough historical data
        if self.n_updates < 1 or len(self.agent2_joint_buffer) < self.min_samples:
            print(f"[HistoricalNormalizer] Not enough agent2 history (updates={self.n_updates}, buffer={len(self.agent2_joint_buffer)}), skipping normalization")
            return agent2_joint_rewards
        
        # Convert to numpy if needed
        is_tensor = isinstance(agent2_joint_rewards, torch.Tensor)
        if is_tensor:
            device = agent2_joint_rewards.device
            dtype = agent2_joint_rewards.dtype
            agent2_joint_rewards_np = agent2_joint_rewards.cpu().numpy()
            agent2_solo_rewards_np = agent2_solo_rewards.cpu().numpy()
        else:
            agent2_joint_rewards_np = np.array(agent2_joint_rewards, dtype=np.float32)
            agent2_solo_rewards_np = np.array(agent2_solo_rewards, dtype=np.float32)
        
        # Step 1: History-normalize both joint and solo rewards
        z_joint = (agent2_joint_rewards_np - self.mu_joint) / (self.sigma_joint + self.epsilon)
        z_solo = (agent2_solo_rewards_np - self.mu_solo) / (self.sigma_solo + self.epsilon)
        
        # Step 2: Compute adaptive gating coefficient
        g = self._sigmoid(self.eta * self.mu_delta / (self.sigma_delta + self.epsilon))
        
        # Step 3: Weighted combination
        r2 = g * z_joint + (1 - g) * z_solo
        
        if is_tensor:
            r2 = torch.tensor(r2, dtype=dtype, device=device)
        
        return r2
    
    def _sigmoid(self, x):
        """Stable sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def get_statistics(self):
        """
        Get current EMA statistics for logging.
        
        Returns:
            Dictionary with statistics
        """
        self._update_statistics()
        
        # Compute gating coefficient for logging
        g = self._sigmoid(self.eta * self.mu_delta / (self.sigma_delta + self.epsilon))
        
        return {
            'agent1/delta_mean': self.mu_delta,
            'agent1/delta_std': self.sigma_delta,
            'agent1/history_size': len(self.agent1_deltas_buffer),
            'agent2/joint_mean': self.mu_joint,
            'agent2/joint_std': self.sigma_joint,
            'agent2/solo_mean': self.mu_solo,
            'agent2/solo_std': self.sigma_solo,
            'agent2/history_size': len(self.agent2_joint_buffer),
            'agent2/gate_coef': g,
            'n_updates': self.n_updates,
        }
    
    def reset(self):
        """Clear all historical data."""
        self.agent1_deltas_buffer.clear()
        self.agent2_joint_buffer.clear()
        self.agent2_solo_buffer.clear()
        
        self.mu_delta = 0.0
        self.sigma_delta = 1.0
        self.m2_delta = 1.0
        
        self.mu_joint = 0.0
        self.sigma_joint = 1.0
        self.m2_joint = 1.0
        
        self.mu_solo = 0.0
        self.sigma_solo = 1.0
        self.m2_solo = 1.0
        
        self.n_updates = 0
        self._stats_dirty = True

