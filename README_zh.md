# 面向多智能体协作的反事实信用策略优化

<p align="center">
  <strong>基于 Thinker-Solver 双模型协作的强化学习训练框架。</strong>
</p>

<p align="center">
  <a href="README.md"><strong>English README</strong></a>
</p>

## 项目概览

本仓库包含两个基于 `verl` 改造的多智能体强化学习训练 recipe，用于训练 Thinker-Solver 协作系统：

- `CCPO`：Counterfactual Credit Policy Optimization，基于反事实结果进行信用分配。
- `SEPO`：基于自评和互评信号进行信用/责任分配的策略优化方法。

两套代码都使用 Thinker + Solver 的两阶段交互结构。Thinker 只根据题目生成解题思路，不直接输出最终答案；Solver 接收原始题目和 Thinker 的思路后生成最终答案。训练时按固定周期交替更新两个角色，例如默认每 10 个 step 切换一次，从训练 Thinker 切换到训练 Solver。

底层优化沿用 `verl` 中的 GRPO/GSPO 训练路径，默认面向 actor-only 训练，并关闭 critic、KL reward 和 KL loss。

## 框架

<p align="center">
  <img src="assets/framework.png" width="100%" alt="反事实信用分配框架"/>
</p>

## 仓库结构

```text
.
├── CCPO/    # 反事实信用分配 recipe
├── SEPO/    # 自评/互评信用分配 recipe
├── assets/  # README 图片和静态资源
└── README.md
```

## 方法简介

### CCPO

`CCPO` 的核心思想是用反事实结果给 Thinker 做信用分配。

每个 prompt 会先采样多个 Thinker thought，然后 Solver 基于每个 thought 生成 joint answer；同时 Solver 还会在没有 thought 的情况下生成 solo answer 作为反事实基线。Verifier 分别判断 joint answer 和 solo answer 是否正确。

- Solver 的奖励主要来自 joint answer 的最终正确性。
- Thinker 的奖励在 joint 正确性基础上加入反事实 bonus。
- 如果有 thought 时答对、没有 thought 时答错，说明 Thinker 的思路有正贡献，Thinker 得到正向 bonus。
- 如果有 thought 时答错、没有 thought 时答对，说明 Thinker 的思路可能误导 Solver，Thinker 得到负向 bonus。

### SEPO

`SEPO` 的核心思想是让两个模型对一次协作进行自评和互评，再把评分作为小幅的信用/责任分配信号。

每个 prompt 同样先由 Thinker 生成 thought，再由 Solver 生成 final answer。Verifier 判断最终答案是否正确后，Thinker 和 Solver 会分别输出一个固定格式的评分 JSON，从固定集合中选择一个二元分配比例，例如 `[0.7, 0.3]`，第一个数表示 Thinker 的贡献或责任，第二个数表示 Solver 的贡献或责任。

训练时会把自评和互评合并：

- `PEER_REWARD_ETA` 控制自评权重，默认 `0.3`；互评权重是 `1 - eta`。
- 答案正确时，评分作为 credit bonus 加到两个角色的奖励上。
- 答案错误时，评分作为 blame signal 调整两个角色的惩罚。
- 默认会按同一 prompt 的 group 做中心化，避免整体奖励尺度漂移。

## 环境安装

两个子目录都是完整的 `verl` 工程。进入需要运行的 recipe 目录后，在该目录中安装环境。

```bash
cd CCPO
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`SEPO` 使用相同的安装方式：

```bash
cd SEPO
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练命令

默认脚本使用 4 张 GPU、vLLM rollout 和离线 Weights & Biases 日志，并假设模型和数据位于 `/gz-data/...`。如果你的路径不同，需要修改脚本中的 `RUN_ROOT`、`train_files`、`val_sets`，或在启动时覆盖模型路径相关环境变量。

### 训练 CCPO

```bash
cd CCPO
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

### 训练 SEPO

```bash
cd SEPO
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

## 常用配置

| 变量 | 说明 |
| --- | --- |
| `THINKER_MODEL_PATH` | Thinker 的 HuggingFace 模型路径。 |
| `SOLVER_MODEL_PATH` | Solver 的 HuggingFace 模型路径。 |
| `ALTERNATE_PERIOD` | 每隔多少个 step 切换一次 Thinker/Solver 训练角色。 |
| `START_WITH` | 初始训练角色，通常为 `thinker`。 |
| `TWO_MODEL_RUNTIME` | 运行模式，例如 `auto`、`checkpoint_swap` 或 `dual_worker`。 |
| `COUNTERFACTUAL_LAMBDA` | CCPO 反事实 bonus 的权重。 |
| `PEER_REWARD_ETA` | SEPO 中自评和互评的混合权重。 |
| `PEER_REWARD_LAMBDA` | SEPO 中正确答案的 credit shaping 强度。 |
| `PEER_BLAME_LAMBDA` | SEPO 中错误答案的 blame shaping 强度。 |

## 备注

- `checkpoint_swap` 适合同一个模型路径的 Thinker/Solver 双角色训练。
- `dual_worker` 支持异构 Thinker 和 Solver 模型，但显存压力更大，必要时需要调小 batch size、rollout 数量、TP 或 response length。
- 当前 recipe 主要面向 actor-only 的 GRPO/GSPO 路径，建议保持 `use_kl_loss=False`、`use_kl_in_reward=False`，除非你明确改造训练流程，否则不要启用 critic/ref policy。

## 致谢

本项目基于以下开源项目构建：

- [ReMA](https://github.com/ziyuwan/ReMA-public)：强化多智能体训练框架。
- [verl](https://github.com/volcengine/verl)：火山引擎大模型强化学习框架。
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)：Qwen2.5 系列模型。
- [vLLM](https://github.com/vllm-project/vllm)：高效大模型推理引擎。


