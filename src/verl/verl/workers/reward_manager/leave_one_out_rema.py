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

from functools import partial
from typing import Dict
import numpy as np

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException

def compute_score_fn(compute_score, params):
    data_source, response, ground_truth, extra_info = params
    return compute_score(data_source, response, ground_truth, extra_info)


def compute_format_r(data_source, role, response_str):
    """Format reward for different data sources and roles"""
    if data_source == "ReMA-math":
        if 'boxed' in response_str:
            if role == 'meta_thinking':
                return -0.25
            elif role == 'reasoning':
                return 0.25
            else:
                raise ValueError(f"Unknown {role=}") 
        else: 
            return 0.0
    elif data_source == 'ReMA-laaj':
        from verl.utils.reward_score.pairwise_laaj import extract_final_verdict
        ans = extract_final_verdict(response_str)
        if ans is not None:
            if role == 'meta_thinking':
                return -0.25
            elif role == 'reasoning':
                return 0.25
            else:
                raise ValueError(f"Unknown {role=}") 
        else: 
            return 0.0
    else:
        raise ValueError(f'Unknown {data_source=} for format reward.')


class LeaveOneOutRewardManager:
    """
    History-Aware Contribution-Based Reward Manager for Dual-Agent Reasoning.
    
    This reward manager implements:
    - Agent1 (meta_thinking): History-aware contribution shaping
      * z^Δ = (Δ - μ_Δ) / (σ_Δ + ε), where Δ = R_joint - R_solo
      * r1 = tanh(α * z^Δ)
      * A1 = GRPO group normalization of r1
      
    - Agent2 (reasoning): History-gated joint-solo optimization
      * z^joint = (R_joint - μ_joint) / (σ_joint + ε)
      * z^solo = (R_solo - μ_solo) / (σ_solo + ε)
      * g = sigmoid(η * μ_Δ / (σ_Δ + ε))
      * r2 = g * z^joint + (1-g) * z^solo
      * A2 = GRPO group normalization of r2
    
    Historical statistics (μ, σ) are maintained using EMA and treated as non-differentiable.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, historical_normalizer=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.historical_normalizer = historical_normalizer

    def __call__(self, data: DataProto, agent2_solo_data: DataProto = None) -> Dict[str, torch.Tensor]:
        """
        Compute rewards with leave-one-out mechanism.
        
        Args:
            data: DataProto containing joint rollouts (agent1 + agent2)
            agent2_solo_data: DataProto containing agent2 solo rollouts (optional)
                             If None, standard reward computation is used
        
        Returns:
            Dictionary mapping agent roles to their turn-level rewards
        """
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']
        agent_roles = data.meta_info['agent_roles']
        
        # Initialize reward tensors
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) 
            for role in agent_roles
        }
        
        # Compute scores for joint rollouts
        joint_scores = self._compute_scores(data)
        
        # If no solo data provided, use standard reward computation
        if agent2_solo_data is None:
            return self._compute_standard_rewards(data, joint_scores, reward_tensor_map)
        
        # Compute scores for agent2 solo rollouts
        agent2_solo_scores = self._compute_scores(agent2_solo_data)
        
        # Implement leave-one-out reward computation
        return self._compute_leave_one_out_rewards(
            data, agent2_solo_data, 
            joint_scores, agent2_solo_scores, 
            reward_tensor_map
        )
    
    def _compute_scores(self, data: DataProto):
        """Compute scores for a batch of data"""
        params = [
            (data[i].non_tensor_batch['data_source'],
             data[i].non_tensor_batch['response'],
             data[i].non_tensor_batch['reward_model']['ground_truth'],
             data[i].non_tensor_batch.get('extra_info', None))
            for i in range(len(data))
        ]
        
        scores = []
        with ProcessPool(max_workers=1) as pool:
            future = pool.map(partial(compute_score_fn, self.compute_score), params, timeout=10)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                while True:
                    try:
                        result = next(iterator)
                        scores.append(result)
                    except TimeoutError:
                        print('Time Out')
                        scores.append(0.0)
                    except TimeoutException:
                        print('Math verify internal timeout')
                        scores.append(0.0)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        raise e
                    pbar.update(1)
        
        return scores
    
    def _compute_standard_rewards(self, data: DataProto, scores, reward_tensor_map):
        """Standard reward computation (no leave-one-out)"""
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']
        agent_roles = data.meta_info['agent_roles']
        already_print_data_sources = {}
        
        accuracy = torch.tensor(scores, dtype=torch.float32)
        reward_tensor_map['acc'] = accuracy
        
        for i_bsz in range(batch_size):
            data_item = data[i_bsz]
            score = scores[i_bsz]
            num_turns = data_item.non_tensor_batch['num_turns']
            data_source = data_item.non_tensor_batch['data_source']
            
            for i_role, role in enumerate(agent_roles):
                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                if data_item.meta_info['mask_unfinished_reward']:
                    score_for_role = score if turn_finished == 0 else 0.0
                else:
                    score_for_role = score
                
                if turn_finished == 0 and data_item.meta_info.get('use_format_reward', False) and max_num_turns == 1:
                    last_round_msg = data_item.non_tensor_batch['history'][i_role]
                    assert last_round_msg['role'] == role, role
                    format_r = compute_format_r(data_source, role, last_round_msg['content'])
                    score_for_role += format_r
                
                reward_tensor_map[f'{role}_turn_level_reward'][i_bsz, num_turns - 1] = score_for_role
            
            # Print debug info
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                self._print_debug_info(data_item, score)
        
        return reward_tensor_map
    
    def _compute_leave_one_out_rewards(
        self, 
        joint_data: DataProto, 
        solo_data: DataProto,
        joint_scores, 
        solo_scores, 
        reward_tensor_map
    ):
        """
        Compute leave-one-out rewards using Shapley values.
        
        Assumption: joint_data and solo_data have the same prompts in the same order.
        Each prompt has n rollouts (e.g., n=4).
        
        Shapley value-based reward allocation:
        - Agent1 (meta_thinking): φ₁ = 0.5 * (joint[k] - solo[k])
          Gets half of the marginal contribution it brings to the coalition.
          
        - Agent2 (reasoning): φ₂ = 0.5 * solo[k] + 0.5 * joint[k]
          Gets half of its standalone value plus half of the joint value.
        
        Efficiency property: φ₁ + φ₂ = joint[k] ✓
        
        Theoretical foundation: Based on Shapley (1953) axiomatic value theory.
        This ensures fair credit assignment that satisfies efficiency, symmetry,
        null player, and additivity axioms.
        """
        batch_size = len(joint_data)
        max_num_turns = joint_data.meta_info['max_num_turns']
        agent_roles = joint_data.meta_info['agent_roles']
        already_print_data_sources = {}
        
        # Group data by prompt UID
        joint_uid_to_indices = {}
        solo_uid_to_indices = {}
        
        for i in range(len(joint_data)):
            uid = joint_data[i].non_tensor_batch['uid']
            if uid not in joint_uid_to_indices:
                joint_uid_to_indices[uid] = []
            joint_uid_to_indices[uid].append(i)
        
        for i in range(len(solo_data)):
            uid = solo_data[i].non_tensor_batch['uid']
            if uid not in solo_uid_to_indices:
                solo_uid_to_indices[uid] = []
            solo_uid_to_indices[uid].append(i)
        
        # Compute rewards for each unique prompt
        all_joint_scores = []
        all_agent1_rewards = []
        all_agent2_rewards = []
        
        # For historical normalization: collect raw deltas, joint rewards, and solo rewards
        raw_agent1_deltas = []
        raw_agent2_joint_rewards = []
        raw_agent2_solo_rewards = []
        
        for uid in joint_uid_to_indices:
            joint_indices = joint_uid_to_indices[uid]
            
            # Get joint scores for this prompt
            joint_scores_this_prompt = [joint_scores[i] for i in joint_indices]
            mean_joint_score = np.mean(joint_scores_this_prompt)
            
            # Get solo scores for this prompt (if available)
            if uid in solo_uid_to_indices:
                solo_indices = solo_uid_to_indices[uid]
                solo_scores_this_prompt = [solo_scores[i] for i in solo_indices]
                
                # Compute raw marginal contributions, joint rewards, and solo rewards (before normalization)
                # Match joint rollout k with solo rollout k
                for k, joint_idx in enumerate(joint_indices):
                    if k < len(solo_scores_this_prompt):
                        # Raw marginal contribution: Δ = R_joint - R_solo
                        marginal_contribution = joint_scores[joint_idx] - solo_scores_this_prompt[k]
                        joint_reward = joint_scores[joint_idx]
                        solo_reward = solo_scores_this_prompt[k]
                    else:
                        # Fallback if solo rollouts are fewer
                        mean_solo = np.mean(solo_scores_this_prompt)
                        marginal_contribution = joint_scores[joint_idx] - mean_solo
                        joint_reward = joint_scores[joint_idx]
                        solo_reward = mean_solo
                    
                    raw_agent1_deltas.append(marginal_contribution)
                    raw_agent2_joint_rewards.append(joint_reward)
                    raw_agent2_solo_rewards.append(solo_reward)
                    # Store tuple of (joint_idx, delta, joint_reward, solo_reward)
                    all_agent1_rewards.append((joint_idx, marginal_contribution))
                    all_agent2_rewards.append((joint_idx, joint_reward, solo_reward))
                    all_joint_scores.append(joint_scores[joint_idx])
            else:
                # No solo data for this prompt, use standard rewards
                for joint_idx in joint_indices:
                    raw_agent1_deltas.append(joint_scores[joint_idx])
                    raw_agent2_joint_rewards.append(joint_scores[joint_idx])
                    raw_agent2_solo_rewards.append(0.0)  # Fallback
                    all_agent1_rewards.append((joint_idx, joint_scores[joint_idx]))
                    all_agent2_rewards.append((joint_idx, joint_scores[joint_idx], 0.0))
                    all_joint_scores.append(joint_scores[joint_idx])
        
        # Apply historical normalization if normalizer is provided
        if self.historical_normalizer is not None:
            print(f"[LeaveOneOut] Applying history-aware contribution-based normalization...")
            print(f"  • Raw agent1 deltas: mean={np.mean(raw_agent1_deltas):.3f}, std={np.std(raw_agent1_deltas):.3f}")
            print(f"  • Raw agent2 joint rewards: mean={np.mean(raw_agent2_joint_rewards):.3f}, std={np.std(raw_agent2_joint_rewards):.3f}")
            print(f"  • Raw agent2 solo rewards: mean={np.mean(raw_agent2_solo_rewards):.3f}, std={np.std(raw_agent2_solo_rewards):.3f}")
            
            # Print OLD historical stats (before update)
            stats_before = self.historical_normalizer.get_statistics()
            print(f"  • Using OLD EMA stats for normalization:")
            print(f"    - agent1 delta: μ={stats_before['agent1/delta_mean']:.3f}, σ={stats_before['agent1/delta_std']:.3f}, n={stats_before['agent1/history_size']}")
            print(f"    - agent2 joint: μ={stats_before['agent2/joint_mean']:.3f}, σ={stats_before['agent2/joint_std']:.3f}")
            print(f"    - agent2 solo: μ={stats_before['agent2/solo_mean']:.3f}, σ={stats_before['agent2/solo_std']:.3f}")
            print(f"    - gate coefficient g={stats_before['agent2/gate_coef']:.3f}")
            
            # Prepare arrays for normalization
            agent1_deltas_array = np.array([delta for _, delta in all_agent1_rewards], dtype=np.float32)
            agent2_joint_array = np.array([joint for _, joint, solo in all_agent2_rewards], dtype=np.float32)
            agent2_solo_array = np.array([solo for _, joint, solo in all_agent2_rewards], dtype=np.float32)
            
            # Apply Agent1 history-aware contribution shaping
            # r1 = tanh(α * z^Δ) where z^Δ = (Δ - μ_Δ) / (σ_Δ + ε)
            shaped_agent1 = self.historical_normalizer.normalize_agent1_rewards(agent1_deltas_array)
            
            # Apply Agent2 history-gated joint-solo optimization
            # r2 = g * z^joint + (1-g) * z^solo
            shaped_agent2 = self.historical_normalizer.normalize_agent2_rewards(agent2_joint_array, agent2_solo_array)
            
            # Update all_agent1_rewards and all_agent2_rewards with shaped values
            all_agent1_rewards = [(idx, float(shaped_agent1[i])) for i, (idx, _) in enumerate(all_agent1_rewards)]
            all_agent2_rewards = [(idx, float(shaped_agent2[i])) for i, (idx, _, _) in enumerate(all_agent2_rewards)]
            
            print(f"  • Shaped agent1 rewards: mean={np.mean(shaped_agent1):.3f}, std={np.std(shaped_agent1):.3f}, range=[{np.min(shaped_agent1):.3f}, {np.max(shaped_agent1):.3f}]")
            print(f"  • Shaped agent2 rewards: mean={np.mean(shaped_agent2):.3f}, std={np.std(shaped_agent2):.3f}, range=[{np.min(shaped_agent2):.3f}, {np.max(shaped_agent2):.3f}]")
            
            # Update historical EMA statistics with current batch (after normalization)
            self.historical_normalizer.update(raw_agent1_deltas, raw_agent2_joint_rewards, raw_agent2_solo_rewards)
            
            # Print NEW historical stats (after update)
            stats_after = self.historical_normalizer.get_statistics()
            print(f"  • Updated EMA stats (after adding current batch):")
            print(f"    - agent1 delta: μ={stats_after['agent1/delta_mean']:.3f}, σ={stats_after['agent1/delta_std']:.3f}, n={stats_after['agent1/history_size']}")
            print(f"    - agent2 joint: μ={stats_after['agent2/joint_mean']:.3f}, σ={stats_after['agent2/joint_std']:.3f}")
            print(f"    - agent2 solo: μ={stats_after['agent2/solo_mean']:.3f}, σ={stats_after['agent2/solo_std']:.3f}")
            print(f"    - gate coefficient g={stats_after['agent2/gate_coef']:.3f}")
            print(f"    - total updates={stats_after['n_updates']}")
        else:
            # No normalization - use original Shapley values (scale by 0.5)
            all_agent1_rewards = [(idx, 0.5 * reward) for idx, reward in all_agent1_rewards]
            all_agent2_rewards = [(idx, 0.5 * reward) for idx, _, _ in all_agent2_rewards]
        
        # Fill reward tensor map
        accuracy = torch.tensor(all_joint_scores, dtype=torch.float32)
        reward_tensor_map['acc'] = accuracy
        
        for joint_idx, agent1_reward in all_agent1_rewards:
            data_item = joint_data[joint_idx]
            num_turns = data_item.non_tensor_batch['num_turns']
            data_source = data_item.non_tensor_batch['data_source']
            
            # Agent1 (meta_thinking) gets leave-one-out reward
            role = 'meta_thinking'
            turn_finished = data_item.batch[f'{role}_turn_finished'].item()
            if data_item.meta_info['mask_unfinished_reward']:
                score_for_agent1 = agent1_reward if turn_finished == 0 else 0.0
            else:
                score_for_agent1 = agent1_reward
            
            reward_tensor_map[f'{role}_turn_level_reward'][joint_idx, num_turns - 1] = score_for_agent1
        
        for joint_idx, agent2_reward in all_agent2_rewards:
            data_item = joint_data[joint_idx]
            num_turns = data_item.non_tensor_batch['num_turns']
            
            # Agent2 (reasoning) gets mean joint reward
            role = 'reasoning'
            turn_finished = data_item.batch[f'{role}_turn_finished'].item()
            if data_item.meta_info['mask_unfinished_reward']:
                score_for_agent2 = agent2_reward if turn_finished == 0 else 0.0
            else:
                score_for_agent2 = agent2_reward
            
            if turn_finished == 0 and data_item.meta_info.get('use_format_reward', False) and max_num_turns == 1:
                last_round_msg = data_item.non_tensor_batch['history'][1]  # reasoning is second
                assert last_round_msg['role'] == role, role
                format_r = compute_format_r(data_source, role, last_round_msg['content'])
                score_for_agent2 += format_r
            
            reward_tensor_map[f'{role}_turn_level_reward'][joint_idx, num_turns - 1] = score_for_agent2
        
        # Print debug info for first few samples
        for joint_idx in range(min(self.num_examine, batch_size)):
            data_item = joint_data[joint_idx]
            data_source = data_item.non_tensor_batch['data_source']
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                agent1_rew = reward_tensor_map['meta_thinking_turn_level_reward'][joint_idx].sum().item()
                agent2_rew = reward_tensor_map['reasoning_turn_level_reward'][joint_idx].sum().item()
                print(f"\n[Leave-One-Out Debug]")
                print(f"[question] {data_item.non_tensor_batch['question']}")
                print(f"[joint_response] {data_item.non_tensor_batch['response']}")
                print(f"[agent1_reward] {agent1_rew:.3f}")
                print(f"[agent2_reward] {agent2_rew:.3f}")
        
        return reward_tensor_map
    
    def _print_debug_info(self, data_item, score):
        """Print debug information"""
        prompt_str = data_item.non_tensor_batch['question']
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        response_str = data_item.non_tensor_batch['response']
        num_turns = data_item.non_tensor_batch['num_turns']
        padded_history = data_item.non_tensor_batch['history']
        history = padded_history[:num_turns * 2]
        
        print("[question]", prompt_str)
        print("[ground_truth]", ground_truth)
        print("[answer]", response_str)
        print("[score]", score)
        print("[history]", history)

