# Counterfactual Credit Policy Optimization for Multi-Agent Collaboration

<p align="center">
  <strong>Two-model collaborative reinforcement learning with Thinker-Solver credit assignment.</strong>
</p>

<p align="center">
  <a href="README_zh.md"><strong>Chinese README</strong></a>
</p>

## Overview

This repository contains two `verl`-based multi-agent reinforcement learning recipes for training a collaborative Thinker-Solver system:

- `CCPO`: Counterfactual Credit Policy Optimization.
- `SEPO`: Self- and peer-evaluation-based policy optimization.

Both recipes follow a two-stage interaction pattern. The Thinker first produces a reasoning thought for the given problem, without directly answering it. The Solver then receives the original problem and the Thinker's thought, and generates the final answer. During training, the two roles are updated alternately at a fixed interval, for example switching every 10 steps from Thinker training to Solver training.

The implementation builds on the GRPO/GSPO training paths in `verl` and is designed for actor-only training by default, with critic, KL reward, and KL loss disabled.

## Framework

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Counterfactual Credit Allocation Framework"/>
</p>

## Repository Layout

```text
.
|-- CCPO/    # Counterfactual credit assignment recipe
|-- SEPO/    # Self- and peer-evaluation credit assignment recipe
|-- assets/  # Figures and README assets
`-- README_zh.md
```

## Methods

### CCPO

`CCPO` assigns credit to the Thinker through counterfactual outcomes.

For each prompt, the system samples multiple Thinker thoughts and asks the Solver to generate a joint answer conditioned on each thought. The Solver also produces a solo answer without any Thinker thought as a counterfactual baseline. A verifier then evaluates both the joint answers and the solo answer.

- The Solver reward mainly comes from the correctness of the joint answer.
- The Thinker reward starts from the joint-answer correctness and adds a counterfactual bonus.
- If the joint answer is correct while the solo answer is wrong, the Thinker likely contributed useful reasoning and receives a positive bonus.
- If the joint answer is wrong while the solo answer is correct, the Thinker may have misled the Solver and receives a negative bonus.

### SEPO

`SEPO` uses self-evaluation and peer evaluation as lightweight credit and responsibility signals.

For each prompt, the Thinker generates a thought and the Solver generates the final answer. After the verifier judges the final answer, both roles output a fixed-format rating JSON. The rating selects a contribution or responsibility split from a predefined set, such as `[0.7, 0.3]`, where the first value corresponds to the Thinker and the second to the Solver.

During training, self-evaluation and peer evaluation are combined:

- `PEER_REWARD_ETA` controls the self-evaluation weight. The default is `0.3`; the peer-evaluation weight is `1 - eta`.
- For correct answers, the rating is used as a credit bonus for the two roles.
- For wrong answers, the rating is used as a blame signal to adjust penalties.
- Group centering is enabled by default to keep the reward scale stable within prompts.

## Installation

Each subdirectory is a complete `verl` project. Enter the recipe directory you want to run and install the environment there.

```bash
cd CCPO
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Use the same installation steps for `SEPO`:

```bash
cd SEPO
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Training

The provided scripts are configured for 4 GPUs, vLLM rollout, offline Weights & Biases logging, and local model/data paths under `/gz-data/...`. If your paths are different, update `RUN_ROOT`, `train_files`, and `val_sets` in the scripts, or override the model path environment variables before launch.

### Train CCPO

```bash
cd CCPO
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

### Train SEPO

```bash
cd SEPO
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

## Common Configuration

| Variable | Description |
| --- | --- |
| `THINKER_MODEL_PATH` | HuggingFace model path for the Thinker. |
| `SOLVER_MODEL_PATH` | HuggingFace model path for the Solver. |
| `ALTERNATE_PERIOD` | Number of steps between Thinker/Solver role switches. |
| `START_WITH` | Initial role to train, usually `thinker`. |
| `TWO_MODEL_RUNTIME` | Runtime mode, such as `auto`, `checkpoint_swap`, or `dual_worker`. |
| `COUNTERFACTUAL_LAMBDA` | Weight of the CCPO counterfactual bonus. |
| `PEER_REWARD_ETA` | Weight used to mix self-evaluation and peer evaluation in SEPO. |
| `PEER_REWARD_LAMBDA` | Credit-shaping strength for correct SEPO answers. |
| `PEER_BLAME_LAMBDA` | Blame-shaping strength for incorrect SEPO answers. |

## Notes

- `checkpoint_swap` is intended for Thinker/Solver training when both roles use the same model path.
- `dual_worker` supports heterogeneous Thinker and Solver models, but requires more GPU memory. You may need to reduce batch size, rollout count, tensor parallel size, or response length.
- These recipes are mainly designed for actor-only GRPO/GSPO training. Keep `use_kl_loss=False`, `use_kl_in_reward=False`, and avoid enabling critic/ref policy unless you intentionally adapt the training setup.

## Acknowledgements

This project is built upon the following open-source projects:

- [ReMA](https://github.com/ziyuwan/ReMA-public): Reinforced Multi-LLM Agents training framework.
- [verl](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLMs.
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5): Qwen2.5 series models.
- [vLLM](https://github.com/vllm-project/vllm): Fast LLM inference engine.

