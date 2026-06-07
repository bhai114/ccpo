# Two-Model Alternate Training (Thinker + Solver)

本目录给出在 `verl-gspo-VD-double` 基础上交替训练两个离线 HuggingFace 大模型的 recipe：

- **Thinker**：根据题目给出解题思路，不给最终答案
- **Solver**：根据题目 + Thinker 的思路，给出最终答案（`\boxed{...}`）

按 `alternate_period`（默认 10）个 step 一组：

- 第 `1..10` step：只更新 Thinker
- 第 `11..20` step：只更新 Solver
- 以此类推

## 运行时模式

脚本通过 `TWO_MODEL_RUNTIME` 控制运行时：

- `auto`：默认。两个模型路径相同时走 `checkpoint_swap`，路径不同时走 `dual_worker`。
- `checkpoint_swap`：单 worker group + 磁盘 checkpoint 切换。只支持 Thinker/Solver 是同一个 HF 模型路径。
- `dual_worker`：Thinker 和 Solver 各自初始化一套 worker/rollout/reward tokenizer，支持 Qwen + Llama 这类不同 tokenizer/不同架构的离线 HF 模型。

`checkpoint_swap` 不能用于 Qwen + Llama。FSDP 模型对象、embedding/vocab、vLLM server 都在初始化时固定，不能靠 `load_checkpoint()` 把一个 Qwen 实例变成 Llama 实例。

## 训练流程

每个训练 step：

```text
原始题目
  -> Thinker.generate(n=rollout.n) 得到 thought_1..thought_n
  -> Solver.generate(每个 thought 生成 1 个 answer)
  -> reward manager 只对 Solver 的 final answer 打分，得到 R in {-1, 1}
  -> Thinker 和 Solver 分别输出评分 JSON，pair 必须来自固定集合
  -> 用 eta=0.3 合并自评/互评，得到 s1/s2，再归一化成 w1/w2
  -> 同一原始题目下对 w1/w2 组内中心化，得到 bonus1/bonus2
  -> R=1 时：r1/r2 = R + peer_reward_lambda * bonus1/bonus2
  -> R=-1 时：r1/r2 = R - peer_blame_lambda * bonus1/bonus2
  -> 当前训练角色是 Thinker：把 r1 放到 thought 的最后一个有效 token
  -> 当前训练角色是 Solver：把 r2 放到 answer 的最后一个有效 token
  -> 按 GSPO/GRPO 更新当前角色
```

现在 reward 不再在 generation 时流式计算，因为 Thinker 输出不是最终答案，而且异构 tokenizer 时会解码错。recipe 会在 Solver 输出后，显式切换到 Solver tokenizer 再计算 reward。
默认参数在脚本里可通过环境变量覆盖：`PEER_REWARD_ETA=0.3`、`PEER_REWARD_LAMBDA=0.2`、`PEER_BLAME_LAMBDA=0.2`、`PEER_CENTER_BY_GROUP=True`、`PEER_RATING_MAX_RESPONSE=512`。默认脚本还会设置 `algorithm.norm_adv_by_std_in_grpo=False`，避免小幅信用/责任分配信号被组内标准差归一化冲淡。

## 验证流程

验证跑完整两阶段 pipeline：

```text
val prompt
  -> Thinker 生成 thought
  -> Solver 基于 prompt + thought 生成 final answer
  -> reward manager 按 metric_data_source 分组统计准确率
```

日志样例会显示 `[Thinker] ... [Solver] ...`，并且 Thinker/Solver 输出分别用自己的 tokenizer 解码。

## 使用方式

同模型双角色（默认会走 `checkpoint_swap`）：

```bash
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

Qwen + Llama 等不同离线 HF 模型（默认 `auto` 会走 `dual_worker`，也可以显式指定）：

```bash
THINKER_MODEL_PATH=/path/to/Qwen3-4B-Base \
SOLVER_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
TWO_MODEL_RUNTIME=dual_worker \
ALTERNATE_PERIOD=5 \
START_WITH=thinker \
bash examples/gspo_trainer/run_qwen3_4b_base_gspo_two_model.sh
```

## 重要限制

- `dual_worker` 会在同一个 Ray resource pool 上 colocate 两套模型。它能支持异构模型，但显存/CPU offload 压力比 `checkpoint_swap` 大，需要根据模型大小调小 batch、TP、response length 或提高 offload。
- `dual_worker` 当前只支持 actor-only 的 GRPO/GSPO 路径；保持 `use_kl_loss=False`、`use_kl_in_reward=False`，不要启用 critic/ref policy。
- Resume 仍未完整实现双角色状态恢复；保存会写出两个 role slot，但标准 `_load_checkpoint()` 还不能恢复完整双模型运行态。
