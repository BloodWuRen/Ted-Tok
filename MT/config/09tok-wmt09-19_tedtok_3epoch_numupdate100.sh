#!/bin/bash
set -euo pipefail

# =========================
# 基础配置
# =========================
NUM_GPUS=8
MASTER_PORT=29505

PROJECT_NAME="streaming_llm"
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
# Streaming / Online 参数
# =========================
ALPHA=0.00005
RECORD_FREQ=10
UPDATE_FREQ=1000
NUM_UPDATE=100
WARM_UP_STEPS=6000
FIX_RATIO=0.7

# =========================
# 阶段定义
# =========================
STAGE="19"
PREV_STAGE="18"

# 当前 19 的新实验名
CURRENT_EXPERIMENT_NAME="09tok_wmt09-19_wsd_online_streaming_${TOTAL_EPOCHS}epoch_numupdate${NUM_UPDATE}"

# =========================
# 工具函数
# =========================

decay_lr() {
    awk "BEGIN {print $1 * $LEARNING_RATE_DECAY_FACTOR}"
}

grow_min_lr_ratio() {
    awk "BEGIN {print $1 / $LEARNING_RATE_DECAY_FACTOR}"
}

# 旧 continual-pretrain 的命名规则：给 18 这种上游模型用
get_base_experiment_name() {
    local stage="$1"
    echo "09tok_wmt${stage}_${TOTAL_EPOCHS}epoch"
}

get_base_output_dir() {
    local stage="$1"
    echo "${BASE_CHECKPOINT_DIR}/$(get_base_experiment_name "$stage")"
}

# 当前 streaming 训练自己的输出目录
get_current_output_dir() {
    echo "${BASE_CHECKPOINT_DIR}/${CURRENT_EXPERIMENT_NAME}"
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
    output_dir="$(get_base_output_dir "$stage")"
    cat "${output_dir}/latest_checkpointed_iteration.txt"
}

get_hf_model_path() {
    local stage="$1"
    local step
    step="$(get_latest_step "$stage")"
    local output_dir
    output_dir="$(get_base_output_dir "$stage")"
    echo "${output_dir}/global_step_${step}/huggingface"
}

run_stage() {
    local stage="$1"
    local base_model_path="$2"
    local lr="$3"
    local min_lr_ratio="$4"
    local is_first_round="$5"
    local is_last_round="$6"

    local experiment_name="${CURRENT_EXPERIMENT_NAME}"
    local output_dir
    output_dir="$(get_current_output_dir)"

    local train_file
    train_file="$(get_train_file "$stage")"

    local val_file
    val_file="$(get_val_file "$stage")"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M)

    local log_file="${LOGS_DIR}/pt_${experiment_name}_${timestamp}.log"

    echo "================================================================="
    echo "Starting streaming training for WMT${stage}"
    echo "Base Model      : ${base_model_path}"
    echo "Learning Rate   : ${lr}"
    echo "Min LR Ratio    : ${min_lr_ratio}"
    echo "Train File      : ${train_file}"
    echo "Val File        : ${val_file}"
    echo "Output Directory: ${output_dir}"
    echo "Experiment Name : ${experiment_name}"
    echo "Is First Round  : ${is_first_round}"
    echo "Is Last Round   : ${is_last_round}"
    echo "Alpha           : ${ALPHA}"
    echo "Record Freq     : ${RECORD_FREQ}"
    echo "Update Freq     : ${UPDATE_FREQ}"
    echo "Num Update      : ${NUM_UPDATE}"
    echo "Warm Up Steps   : ${WARM_UP_STEPS}"
    echo "Fix Ratio       : ${FIX_RATIO}"
    echo "================================================================="

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        -m verl.trainer.fsdp_sft_streaming_trainer \
        model.strategy="fsdp" \
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
        optim.weight_decay="${WEIGHT_DECAY}" \
        optim.lr_scheduler="multiround_wsd" \
        optim.min_lr_ratio="${min_lr_ratio}" \
        optim.is_first_round="${is_first_round}" \
        optim.is_last_round="${is_last_round}" \
        trainer.logger='["tensorboard"]' \
        algorithm.update_freq="${UPDATE_FREQ}" \
        algorithm.record_freq="${RECORD_FREQ}" \
        algorithm.num_update="${NUM_UPDATE}" \
        algorithm.fix_ratio="${FIX_RATIO}" \
        algorithm.warm_up_steps="${WARM_UP_STEPS}" \
        algorithm.alpha="${ALPHA}" \
        2>&1 | tee "${log_file}"

    python ./verl/scripts/Tedtok_model_merger.py merge \
        --local_dir "${output_dir}/global_step_${latest_step}" \
        --tokenizer_path "${output_dir}/global_step_${latest_step}/tokenizer" \
        --hf_model_config_path "${base_model_path}" \
        --target_dir "${output_dir}/global_step_${latest_step}/huggingface"
}

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

# 从旧命名规则下的 18 latest ckpt 读取 base model
CURRENT_MODEL_PATH="$(get_hf_model_path "${PREV_STAGE}")"

# =========================
# 执行当前 stage
# =========================

echo "Base model used   : ${CURRENT_MODEL_PATH}"
echo "Current LR        : ${CURRENT_LEARNING_RATE}"
echo "Current Min LR    : ${CURRENT_MIN_LR_RATIO}"
echo "Current Exp Name  : ${CURRENT_EXPERIMENT_NAME}"

run_stage \
    "${STAGE}" \
    "${CURRENT_MODEL_PATH}" \
    "${CURRENT_LEARNING_RATE}" \
    "${CURRENT_MIN_LR_RATIO}" \
    "false" \
    "true"
