# CCPO: Counterfactual Credit Policy Optimization for Multi-Agent Collaboration

[**中文版 README**](README_zh.md)



Collaborative multi-agent large language models (LLMs) can solve complex reasoning tasks by decomposing roles and aggregating diverse hypotheses. Yet, reinforcement learning (RL) for such systems is often undermined by credit assignment: a shared global reward obscures individual contributions, inflating update variance and encouraging free-riding. We introduce **Counterfactual Credit Policy Optimization (CCPO)**, a framework that assigns agent-specific learning signals by estimating each agent's marginal contribution through counterfactual trajectories. CCPO builds dynamic counterfactual baselines that simulate outcomes with an agent's contribution removed, yielding role-sensitive advantages for policy optimization. To further improve stability under heterogeneous tasks and data distributions, we propose a **global-history-aware normalization** scheme that calibrates advantages using global rollout statistics. We evaluate CCPO on two collaboration topologies: a sequential Think–Reason dyad and multi-agent voting. Across mathematical and logical reasoning benchmarks, CCPO mitigates free-riding and outperforms strong multi-agent RL baselines, yielding finer-grained and more effective credit assignment for collaborative LLM training.

## Framework

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Counterfactual Credit Allocation Framework"/>
</p>



## Project Structure

```
.
├── assets/                          # Figures and diagrams
├── data/                            # Training and evaluation datasets
│   ├── MATH/                        # MATH dataset (train/test splits)
│   ├── aime2024_fixed_with_prompt.parquet
│   ├── aime2025_fixed_with_prompt.parquet
│   └── ...
├── prompt/                          # Prompt templates for math reasoning
├── scripts/
│   ├── rl/separated_rema/           # RL training scripts
│   ├── sft/                         # Supervised fine-tuning scripts
│   ├── eval/                        # Evaluation scripts
│   └── deepspeed/                   # DeepSpeed config
├── src/
│   ├── verl/                        # Core training framework (based on verl)
│   │   └── verl/
│   │       ├── rema_separated_trainer/  # Dual-agent separated trainer
│   │       │   ├── main_ppo.py          # Main entry point
│   │       │   └── ppo/
│   │       │       ├── ray_trainer.py   # RayReMASeparatedTrainer
│   │       │       └── multi_agent_rollout.py
│   │       └── workers/
│   │           └── reward_manager/
│   │               ├── leave_one_out_rema.py    # Leave-one-out reward
│   │               └── historical_normalizer.py # EMA normalization
│   └── 360-LLaMA-Factory/          # LLaMA-Factory for SFT
├── requirements.txt
└── README.md
```

## Installation

**Prerequisites**: Python 3.12, CUDA 12.4

### 1. Install Flash Attention

```bash
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 2. Install verl (editable mode)

```bash
cd src/verl
pip install -e .
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Run the example training script with Qwen2.5-7B on MATH-500:

```bash
bash scripts/rl/separated_rema/example_test500_qwen2_5_7b.sh
```

Before running, modify the script to set your local paths:

```bash
# Set your working directory
cd /path/to/your/project

# Set model paths (can be the same model for both agents)
MODEL_PATH_1="/path/to/Qwen2.5-7B-Instruct"
MODEL_PATH_2="/path/to/Qwen2.5-7B-Instruct"
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `reward_model.reward_manager` | Reward manager type | `leave_one_out_rema` |
| `reward_model.use_historical_normalization` | Enable EMA normalization | `True` |
| `reward_model.ema_decay` | EMA decay factor | `0.99` |
| `reward_model.alpha` | Scaling factor for Agent 1 reward | `1.0` |
| `reward_model.eta` | Gate control for Agent 2 reward | `1.0` |
| `algorithm.switch_agent.enable` | Enable alternating agent training | `True` |
| `algorithm.switch_agent.freq` | Switch frequency (steps) | `10` |
| `algorithm.adv_estimator` | Advantage estimator | `grpo` |
| `actor_rollout_ref.rollout.n` | Number of rollout samples | `4` |
| `actor_rollout_ref.actor.clip_mode` | PPO clipping mode | `turn` |
| `trainer.total_epochs` | Total training epochs | `10` |

## Acknowledgements

This project is built upon the following open-source projects:

- [ReMA](https://github.com/ziyuwan/ReMA-public) - Reinforced Multi-LLM Agents training framework
- [verl](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - Qwen2.5 series models
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference engine

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{ccpo2025,
  title={Counterfactual Credit Policy Optimization for Multi-Agent Collaboration},
  year={2025}
}
```
