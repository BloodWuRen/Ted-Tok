#!/bin/bash
set -euo pipefail

# =========================
# 基础配置
# =========================
NUM_GPUS=8
MASTER_PORT=29505

PROJECT_NAME="ted_tok_continual_pt"
BASE_DATA_DIR="./data"
BASE_CHECKPOINT_DIR="./checkpoints/models"
LOGS_DIR="logs"
mkdir -p "${LOGS_DIR}"

TOTAL_EPOCHS=3
MAX_LENGTH=1024
TRUNCATION="left"
PROMPT_KEY="en"
RESPONSE_KEY="de"
TRAIN_BATCH_SIZE=576
MICRO_BATCH_SIZE=24
SAVE_FREQ=2000
TEST_FREQ=200
WEIGHT_DECAY=1e-1

INITIAL_LEARNING_RATE=6e-4
INITIAL_MIN_LR_RATIO=0.1
LEARNING_RATE_DECAY_FACTOR=0.85

CUDA_DEVICES="0,1,2,3,4,5,6,7"

# =========================
# 工具函数
# =========================

decay_lr() {
    awk "BEGIN {print $1 * $LEARNING_RATE_DECAY_FACTOR}"
}

grow_min_lr_ratio() {
    awk "BEGIN {print $1 / $LEARNING_RATE_DECAY_FACTOR}"
}

get_experiment_name() {
    local stage="$1"
    echo "09tok_wmt${stage}_${TOTAL_EPOCHS}epoch"
}

get_output_dir() {
    local stage="$1"
    echo "${BASE_CHECKPOINT_DIR}/$(get_experiment_name "$stage")"
}

get_train_file() {
    local stage="$1"
    echo "${BASE_DATA_DIR}/wmt${stage}/de-en/wmt${stage}_en_de_decoder_only_dataset/train/train.parquet"
}

get_val_file() {
    local stage="$1"
    echo "${BASE_DATA_DIR}/wmt${stage}/de-en/wmt${stage}_en_de_decoder_only_dataset/validation/validation.parquet"
}

get_latest_step() {
    local stage="$1"
    local output_dir
    output_dir="$(get_output_dir "$stage")"
    cat "${output_dir}/latest_checkpointed_iteration.txt"
}

get_hf_model_path() {
    local stage="$1"
    local step
    step="$(get_latest_step "$stage")"
    local output_dir
    output_dir="$(get_output_dir "$stage")"
    echo "${output_dir}/global_step_${step}/huggingface"
}

run_stage() {
    local stage="$1"
    local base_model_path="$2"
    local lr="$3"
    local min_lr_ratio="$4"
    local is_first_round="$5"
    local is_last_round="$6"

    local experiment_name
    experiment_name="$(get_experiment_name "$stage")"

    local output_dir
    output_dir="$(get_output_dir "$stage")"

    local train_file
    train_file="$(get_train_file "$stage")"

    local val_file
    val_file="$(get_val_file "$stage")"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M)

    local log_file="${LOGS_DIR}/pt_${experiment_name}_${timestamp}.log"

    echo "================================================================="
    echo "Starting training for WMT${stage}"
    echo "Base Model      : ${base_model_path}"
    echo "Learning Rate   : ${lr}"
    echo "Min LR Ratio    : ${min_lr_ratio}"
    echo "Train File      : ${train_file}"
    echo "Val File        : ${val_file}"
    echo "Output Directory: ${output_dir}"
    echo "Is First Round  : ${is_first_round}"
    echo "Is Last Round   : ${is_last_round}"
    echo "================================================================="

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="${train_file}" \
        data.val_files="${val_file}" \
        data.prompt_key="${PROMPT_KEY}" \
        data.response_key="${RESPONSE_KEY}" \
        data.max_length="${MAX_LENGTH}" \
        data.truncation="${TRUNCATION}" \
        data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        model.partial_pretrain="${base_model_path}" \
        trainer.default_local_dir="${output_dir}" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        optim.lr="${lr}" \
        optim.lr_scheduler="multiround_wsd" \
        optim.min_lr_ratio="${min_lr_ratio}" \
        optim.is_first_round="${is_first_round}" \
        optim.is_last_round="${is_last_round}" \
        optim.weight_decay="${WEIGHT_DECAY}" \
        trainer.logger='["tensorboard"]' \
        2>&1 | tee "${log_file}"

    echo "Finding the latest checkpoint and merging..."
    local latest_step
    latest_step="$(get_latest_step "$stage")"

    python ./verl/scripts/Tedtok_model_merger.py merge \
        --local_dir "${output_dir}/global_step_${latest_step}" \
        --hf_model_config_path "${base_model_path}" \
        --target_dir "${output_dir}/global_step_${latest_step}/huggingface"

    local next_model_path
    next_model_path="${output_dir}/global_step_${latest_step}/huggingface"

    echo "${next_model_path}"
}

# =========================
# 阶段定义
# =========================

stage="19"

# =========================
# 初始化 base model / lr
# =========================

CURRENT_LEARNING_RATE="${INITIAL_LEARNING_RATE}"
CURRENT_MIN_LR_RATIO="${INITIAL_MIN_LR_RATIO}"

NUM_DECAY_STEPS=7
for ((i=0; i<NUM_DECAY_STEPS; i++)); do
    CURRENT_LEARNING_RATE=$(decay_lr "${CURRENT_LEARNING_RATE}")
    CURRENT_MIN_LR_RATIO=$(grow_min_lr_ratio "${CURRENT_MIN_LR_RATIO}")
done

CURRENT_MODEL_PATH="$(get_hf_model_path "18")"

NEXT_MODEL_PATH=$(
    run_stage \
        "${stage}" \
        "${CURRENT_MODEL_PATH}" \
        "${CURRENT_LEARNING_RATE}" \
        "${CURRENT_MIN_LR_RATIO}" \
        "false" \
        "true"
)