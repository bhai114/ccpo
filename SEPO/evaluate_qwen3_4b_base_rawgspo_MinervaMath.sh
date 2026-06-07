#!/usr/bin/env bash
set -xeuo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# 用法示例：
# 1) 默认评测 best_model：
#    bash examples/gspo_trainer/evaluate_qwen3_4b_base_rawgspo_AIME24.sh
# 2) 修改 best_model 路径：
#    BEST_MODEL_PATH=/path/to/best_model bash examples/gspo_trainer/evaluate_qwen3_4b_base_rawgspo_AIME24.sh
# 3) 直接传 actor checkpoint：
#    MODEL_PATH=/path/to/best_model/actor bash examples/gspo_trainer/evaluate_qwen3_4b_base_rawgspo_AIME24.sh
# 4) 直接传已合并好的 HuggingFace 模型：
#    MODEL_PATH=/path/to/merged_hf_model bash examples/gspo_trainer/evaluate_qwen3_4b_base_rawgspo_AIME24.sh

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
环境变量：
  REPO_ROOT                     verl-gspo-VD-double 仓库路径
  DATA_PATH                     待评测 parquet 数据集路径
  BEST_MODEL_PATH               直接指向 best_model 目录
  MODEL_PATH                    推理模型目录；默认继承 BEST_MODEL_PATH
  CKPT_BACKEND                  checkpoint 后端，默认 fsdp
  MERGED_MODEL_DIR              合并后 HuggingFace 模型保存目录
  OUTPUT_PATH                   生成结果 parquet 路径
  NNODES                        推理节点数，默认 1
  NGPUS_PER_NODE                每个节点 GPU 数，默认 1
  GEN_TP                        推理 tensor parallel size，默认 1
  PIPELINE_MODEL_PARALLEL_SIZE  推理 pipeline parallel size，默认 1
  BATCH_SIZE                    推理 batch size，默认 32
  TEMPERATURE                   采样温度，默认 0.0（贪婪解码）
  TOP_K                         top-k，默认 -1
  TOP_P                         top-p，默认 1.0
  PROMPT_LENGTH                 最大 prompt 长度，默认 2048
  RESPONSE_LENGTH               最大 response 长度，默认 4096
  GPU_MEMORY_UTILIZATION        vLLM 显存利用率，默认 0.6
  TRUST_REMOTE_CODE             是否信任远端代码，默认 False
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ROOT="${REPO_ROOT:-/gz-data/verl-gspo-raw-save}"
DATA_PATH="${DATA_PATH:-/gz-data/verl-main/data/MinervaMath_with_prompt.parquet}"

BEST_MODEL_PATH="${BEST_MODEL_PATH:-$REPO_ROOT/checkpoint/verl_grpo_math/qwen3-4b-base-math500-gspo-fsdp/best_model}"
MODEL_PATH="${MODEL_PATH:-$BEST_MODEL_PATH}"
CKPT_BACKEND="${CKPT_BACKEND:-fsdp}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
GEN_TP="${GEN_TP:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:--1}"
TOP_P="${TOP_P:-1.0}"
PROMPT_LENGTH="${PROMPT_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

[[ -f "$DATA_PATH" ]] || die "DATA_PATH 不存在：$DATA_PATH"

TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
INFER_WORLD_SIZE=$((GEN_TP * PIPELINE_MODEL_PARALLEL_SIZE))
(( INFER_WORLD_SIZE <= TOTAL_GPUS )) || die "GEN_TP * PIPELINE_MODEL_PARALLEL_SIZE = $INFER_WORLD_SIZE，超过可用 GPU 数 $TOTAL_GPUS"

ACTOR_CKPT_DIR=""
MODEL_ROOT=""

normalize_model_or_ckpt_path() {
    [[ -n "$MODEL_PATH" ]] || die "MODEL_PATH 为空"

    # 已合并的 HuggingFace 模型（含 config.json）
    if [[ -f "$MODEL_PATH/config.json" ]]; then
        MODEL_ROOT="$MODEL_PATH"
        return
    fi

    # actor checkpoint 目录（FSDP 或 huggingface 格式）
    if [[ -f "$MODEL_PATH/fsdp_config.json" || -d "$MODEL_PATH/huggingface" ]]; then
        ACTOR_CKPT_DIR="$MODEL_PATH"
        MODEL_ROOT="$(dirname "$MODEL_PATH")"
        MODEL_PATH=""
        return
    fi

    # best_model 目录（包含 actor 子目录）
    if [[ -d "$MODEL_PATH/actor" ]] && [[ -f "$MODEL_PATH/actor/fsdp_config.json" || -d "$MODEL_PATH/actor/huggingface" ]]; then
        MODEL_ROOT="$MODEL_PATH"
        ACTOR_CKPT_DIR="$MODEL_PATH/actor"
        MODEL_PATH=""
        return
    fi

    die "MODEL_PATH 既不是 best_model 目录，也不是 actor checkpoint 目录，或已合并的 HuggingFace 模型目录：$MODEL_PATH"
}

normalize_model_or_ckpt_path

if [[ -z "$MODEL_PATH" ]]; then
    [[ -d "$MODEL_ROOT" ]] || die "模型根目录不存在：$MODEL_ROOT"
    [[ -d "$ACTOR_CKPT_DIR" ]] || die "actor checkpoint 目录不存在：$ACTOR_CKPT_DIR"
    [[ -f "$ACTOR_CKPT_DIR/fsdp_config.json" || -d "$ACTOR_CKPT_DIR/huggingface" ]] || die "actor checkpoint 目录格式不符合预期：$ACTOR_CKPT_DIR"

    MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$MODEL_ROOT/actor_merged_hf}"
    OUTPUT_PATH="${OUTPUT_PATH:-$MODEL_ROOT/minervaMath_qwen3_4b_base_rawgspo_eval.parquet}"

    if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
        python3 -m verl.model_merger merge \
            --backend "$CKPT_BACKEND" \
            --local_dir "$ACTOR_CKPT_DIR" \
            --target_dir "$MERGED_MODEL_DIR"
    fi

    MODEL_PATH="$MERGED_MODEL_DIR"
else
    OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/data/minervaMath_qwen3_4b_base_rawgspo_eval.parquet}"
fi

[[ -f "$MODEL_PATH/config.json" ]] || die "MODEL_PATH 不是有效的 HuggingFace 模型目录：$MODEL_PATH"

echo "=========================================="
echo "Resolved model path : $MODEL_PATH"
echo "Evaluation dataset  : $DATA_PATH"
echo "Generation output   : $OUTPUT_PATH"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRUST_FLAG=""
if [[ "$TRUST_REMOTE_CODE" == "True" || "$TRUST_REMOTE_CODE" == "true" ]]; then
    TRUST_FLAG="--trust_remote_code"
fi

python3 "$SCRIPT_DIR/eval_gaokao2023.py" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --tensor_parallel_size "$GEN_TP" \
    --pipeline_parallel_size "$PIPELINE_MODEL_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_prompt_length "$PROMPT_LENGTH" \
    --max_response_length "$RESPONSE_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    $TRUST_FLAG
