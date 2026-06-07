#!/usr/bin/env bash
# Two-model alternate training (Thinker + Solver), based on Olmo-3-7B-Instruct + Qwen2.5-1.5B-Instruct
# - Thinker:   given a problem, outputs *thought* only (no final answer)
# - Solver:    given a problem + thought, outputs the final answer
# - Training:  every `alternate_period` steps we swap which model is being
#              trained. If model paths differ, the recipe uses dual_worker
#              so Qwen/Llama-style offline HF models keep separate workers.
#
# This script mirrors `run_qwen2_5_1_5b_gspo.sh` and only
# diverges where the two-model recipe requires.

set -x
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Disable robust mean (match the plainmean A/B baseline)
export VERL_USE_ROBUST_MEAN_VD=0
export VERL_USE_ROBUST_MEAN_VD_ADV=0
export VERL_SAVE_GSPO_KL_LOGS=0
export VERL_REWARD_VERBOSE_TIMEOUTS=0
export VERL_TIMEOUT_VERBOSE=0

# ============================================================================
# Paths / PYTHONPATH
# ============================================================================
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CODE_ROOT=${CODE_ROOT:-$SCRIPT_REPO_ROOT}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals}
cd "$RUN_ROOT"
export PYTHONPATH=$CODE_ROOT:$CODE_ROOT/src/verl:$PYTHONPATH
# Make the new recipe package importable. Keep it from the same code tree
# as `verl`, otherwise framework patches in this recipe are silently ignored.
RECIPE_DIR=$CODE_ROOT/examples/gspo_trainer/two_model
export PYTHONPATH=$RECIPE_DIR:$PYTHONPATH

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# Fixed: single node, 4 GPUs
ARNOLD_WORKER_GPU=4
ARNOLD_WORKER_NUM=1

# wandb
backend=fsdp
project_name=ccpo-gspo_TWO_MODEL
experiment_name=olmo3-7b-qwen2.5-1.5b-instruct-math500-twomodel-confact-$backend
default_local_dir=$DATA_ROOT/checkpoint/$project_name/$experiment_name

# ============================================================================
# Two-model specific knobs
#
# Configure the offline HuggingFace model directories here so the script can be
# launched directly without exporting env vars first. You can still override any
# of these from the shell when running quick experiments.
# ============================================================================
: "${THINKER_MODEL_PATH:=/root/autodl-tmp/models/Olmo-3-7B-Instruct}"
: "${SOLVER_MODEL_PATH:=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
: "${ALTERNATE_PERIOD:=10}"        # 1..N -> thinker, N+1..2N -> solver, ...
: "${START_WITH:=thinker}"
: "${TWO_MODEL_RUNTIME:=dual_worker}" # auto | checkpoint_swap | dual_worker
: "${COUNTERFACTUAL_LAMBDA:=0.2}"  # r_thinker = R_joint + lambda * counterfactual_bonus
: "${COUNTERFACTUAL_USE_GROUP_SOLO_BASELINE:=True}"
: "${COUNTERFACTUAL_USE_EVENT_BONUS:=True}"
: "${NORM_ADV_BY_STD_IN_GRPO:=False}"

thinker_model_path=$THINKER_MODEL_PATH
solver_model_path=$SOLVER_MODEL_PATH
alternate_period=$ALTERNATE_PERIOD
start_with=$START_WITH
two_model_runtime=$TWO_MODEL_RUNTIME
counterfactual_lambda=$COUNTERFACTUAL_LAMBDA
counterfactual_use_group_solo_baseline=$COUNTERFACTUAL_USE_GROUP_SOLO_BASELINE
counterfactual_use_event_bonus=$COUNTERFACTUAL_USE_EVENT_BONUS
norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO
swap_dir=$DATA_ROOT/checkpoint/$project_name/$experiment_name/_swap
trust_remote_code=True

# Per-role max response length (thinker is shorter than solver because
# it produces only the thought)
thinker_max_response=$((1024 * 4))
solver_max_response=$((1024 * 6))

# ============================================================================
# Algorithm
# ============================================================================
adv_estimator=grpo
loss_mode=gspo

use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=3e-4
clip_ratio_high=4e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ============================================================================
# Data / Model
# ============================================================================
train_files=/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/MATH/train.parquet
VAL_VIEW_DIR=$DATA_ROOT/data/val_metric_views/olmo3_7b_qwen2_5_1_5b_gspo_twomodel

val_files=$(
    VAL_VIEW_DIR="$VAL_VIEW_DIR" python3 - <<'PY'
import json
import os
import numbers
from pathlib import Path

import pandas as pd

val_sets = [
    ("math500", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/MATH/test500.parquet"),
    ("aime2024", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/aime2024_fixed_with_prompt.parquet"),
    ("aime2025", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/aime2025_fixed_with_prompt.parquet"),
    ("amc23", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/amc23_fixed_with_prompt.parquet"),
    ("gaokao2023", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/Gaokao2023-Math-En_with_prompt.parquet"),
    ("minervamath", "/root/autodl-tmp/ccpo-verl-mulitAgent-Counterfactuals/data/MinervaMath_with_prompt.parquet"),
]

view_dir = Path(os.environ["VAL_VIEW_DIR"])
view_dir.mkdir(parents=True, exist_ok=True)


def normalize_answer(value):
    if pd.isna(value):
        return None
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


view_paths = []
for metric_data_source, src_path in val_sets:
    df = pd.read_parquet(src_path)
    if "answer" in df.columns:
        df["answer"] = df["answer"].map(normalize_answer)
    df["metric_data_source"] = metric_data_source
    dst_path = view_dir / f"{metric_data_source}.parquet"
    df.to_parquet(dst_path, index=False)
    view_paths.append(str(dst_path))

print(json.dumps(view_paths))
PY
)

# Initial actor path (matters only for the worker_group's first init;
# we'll swap to thinker / solver during training).
actor_model_path=$thinker_model_path
critic_model_path=$actor_model_path

# Max prompt length must accommodate (problem + thinker_thought) for the
# solver stage, so we grow it relative to the single-model script.
max_prompt_length=$((1024 * 4))
# `max_response_length` is dynamically overridden per role inside the trainer
max_response_length=$solver_max_response
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

train_batch_size=192
ppo_mini_batch_size=24
n_resp_per_prompt=16
n_resp_per_prompt_val=1
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-4}

# ============================================================================
# Token-length budgets
# ============================================================================
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3))
critic_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4))
infer_max_token_len_per_gpu=$actor_max_token_len_per_gpu

# FSDP parallelism config
USP_SIZE=1
ACTOR_FSDP_CONFIG="
    actor_rollout_ref.actor.fsdp_config.strategy=$backend \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$USP_SIZE"

# Actor config (note: model.path is the *initial* model; we hot-swap later)
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.trust_remote_code=$trust_remote_code \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

CRITIC_CONFIG=""

if [[ $backend == "megatron" ]]; then
    CONFIG_NAME=ppo_megatron_trainer
else
    CONFIG_NAME=ppo_trainer
    ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_FSDP_CONFIG"
fi

# ============================================================================
# Inference (vLLM)
# ============================================================================
rollout_name=vllm
if [ "$rollout_name" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
infer_tp=2
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.7   # leave some headroom for swap

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=$max_prompt_length \
    actor_rollout_ref.rollout.response_length=$max_response_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$infer_max_token_len_per_gpu \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.agent.num_workers=$agent_loop_num_workers \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$infer_max_token_len_per_gpu"

REWARD_CONFIG="
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length}"

# ============================================================================
# Two-model knobs (injected at the top-level of the OmegaConf config)
# ============================================================================
TWO_MODEL_CONFIG="
    +two_model.thinker_model_path=$thinker_model_path \
    +two_model.solver_model_path=$solver_model_path \
    +two_model.alternate_period=$alternate_period \
    +two_model.thinker_max_response=$thinker_max_response \
    +two_model.solver_max_response=$solver_max_response \
    +two_model.swap_dir=$swap_dir \
    +two_model.start_with=$start_with \
    +two_model.runtime=$two_model_runtime \
    +two_model.counterfactual_lambda=$counterfactual_lambda \
    +two_model.counterfactual_use_group_solo_baseline=$counterfactual_use_group_solo_baseline \
    +two_model.counterfactual_use_event_bonus=$counterfactual_use_event_bonus"

# ============================================================================
# Run
# ============================================================================
python3 -m main_two_model \
    --config-path=$CODE_ROOT/verl/trainer/config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    data.train_files="$train_files" \
    "data.val_files=$val_files" \
    data.trust_remote_code=$trust_remote_code \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=$ARNOLD_WORKER_GPU \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=-1 \
    +trainer.save_best_model=False \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    $ACTOR_CONFIG \
    $CRITIC_CONFIG \
    $ROLLOUT_CONFIG \
    $REWARD_CONFIG \
    $TWO_MODEL_CONFIG
