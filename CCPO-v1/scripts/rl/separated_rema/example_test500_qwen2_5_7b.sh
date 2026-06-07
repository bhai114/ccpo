echo $(which python)

# 使用 HuggingFace 镜像站加速下载
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline

# 确保使用正确的代码路径
cd /gz-data/leaveOneOutRwdC_method3_1_3
export PYTHONPATH=/gz-data/leaveOneOutRwdC_method3_1_3/src/verl:$PYTHONPATH

WORKSPACE="./"
PROJECT_NAME="ccpo_leave_one_out"
EXPERIMENT_NAME="qwen2.5-7b-LeaveOneOutRwd-math500-geek1-method3_1_3_savecheckpoints"
MODEL_PATH_1="/gz-data/models/Qwen2.5-7B-Instruct"
MODEL_PATH_2="/gz-data/models/Qwen2.5-7B-Instruct"


python -m verl.rema_separated_trainer.main_ppo \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.train_files=data/MATH/train.parquet \
    data.val_files=data/MATH/test500.parquet \
    data.val_batch_size=64 \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    algorithm.switch_agent.enable=True \
    algorithm.switch_agent.freq=10 \
    algorithm.switch_agent.model_paths=[$MODEL_PATH_1,$MODEL_PATH_2] \
    actor_rollout_ref.model.path=$MODEL_PATH_1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.clip_mode=turn \
    actor_rollout_ref.actor.agg_mode=trajectory \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_turns=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.stop_when_truncated=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    +trainer.val_before_train=True \
    +trainer.val_only=False \
    +trainer.save_val_generations=True \
    +trainer.save_train_generations=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.total_epochs=10 \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=leave_one_out_rema \
    reward_model.mask_unfinished_reward=True \
    reward_model.use_historical_normalization=True \
    reward_model.historical_buffer_size=1000 \
    reward_model.historical_min_samples=10 \
    +reward_model.ema_decay=0.99 \
    +reward_model.alpha=1.0 \
    +reward_model.eta=1.0 \
    algorithm.filter_groups.enable=False \
    trainer.logger=[console,wandb]
