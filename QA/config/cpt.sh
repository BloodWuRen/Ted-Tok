#!/bin/bash
set -euo pipefail

DATA_PATH="./data/WMT_rawtext"
TOKENIZERS_PATH="./checkpoints/tokenizers"
OUTPUT_PATH="./checkpoints/models"
TRAIN_START_YEAR=2007
TRAIN_END_YEAR=2021

TOK_START_YEAR=2007
TOK_END_YEAR=2009

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --tokenizers_path)
      TOKENIZERS_PATH="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --train_start_year)
      TRAIN_START_YEAR="$2"
      shift 2
      ;;
    --train_end_year)
      TRAIN_END_YEAR="$2"
      shift 2
      ;;
    --tok_start_year)
      TOK_START_YEAR="$2"
      shift 2
      ;;
    --tok_end_year)
      TOK_END_YEAR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--data_path PATH] [--tokenizers_path PATH] [--output_path PATH] [--train_start_year YEAR] [--train_end_year YEAR] [--tok_start_year YEAR] [--tok_end_year YEAR]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

build_data_dirs() {
  local data_path="$1"
  local start_year="$2"
  local end_year="$3"

  local cur_start cur_end stage_name full_path
  DATA_DIRS=()

  cur_start="$start_year"
  while [[ "$cur_start" -le "$end_year" ]]; do
    cur_end=$((cur_start + 2))
    if [[ "$cur_end" -gt "$end_year" ]]; then
      cur_end="$end_year"
    fi

    full_path="${data_path}/wmt_${cur_start}-${cur_end}"
    DATA_DIRS+=("$full_path")

    cur_start=$((cur_start + 3))
  done
}

build_data_dirs "$DATA_PATH" "$TRAIN_START_YEAR" "$TRAIN_END_YEAR"

echo "DATA_DIRS:"
for dir in "${DATA_DIRS[@]}"; do
  echo "  $dir"
done

MODEL_TYPE=small

BATCH_SIZE=36
SAVE_INTERVAL=2000
UPDATE_INTERVAL=-1
RECORD_INTERVAL=-1
NUM_UPDATE=100
FIX_RATIO=0.7
NUM_EPOCHS=10
WARMUP_RATIO=0.04

LR_INIT=0.0006
LR_DECAY=0.85
CONST_MIN_LR=0.00006

mkdir -p logs

NUM_STAGES=${#DATA_DIRS[@]}
PREV_OUT_DIR=""

run_stage () {
  local data_dir="$1"
  local lr="$2"
  local min_lr="$3"
  local init_from="$4"
  local base_dir="$5"
  local base_lr="$6"

  local data_name
  data_name=$(basename "${data_dir}")

  local start_year="${TOK_START_YEAR}"
  local end_year="${TOK_END_YEAR}"

  local out_dir="${OUTPUT_PATH}/ours-${MODEL_TYPE}-${NUM_EPOCHS}epoch_${init_from}_${data_name}_tok${start_year}-${end_year}"
  if [[ "${UPDATE_INTERVAL}" -gt 0 ]]; then
    out_dir="${out_dir}_record-interval${RECORD_INTERVAL}_update-interval${UPDATE_INTERVAL}_fix-ratio${FIX_RATIO}"
  fi

  local log_file="logs/train_${data_name}_tok${start_year}-${end_year}.log"

  echo "=============================="
  echo "DATA_DIR    = ${data_dir}"
  echo "INIT_FROM   = ${init_from}"
  echo "BASE_DIR    = ${base_dir}"
  echo "OUT_DIR     = ${out_dir}"
  echo "LR          = ${lr}"
  echo "MIN_LR      = ${min_lr}"
  echo "BASE_LR     = ${base_lr}"
  echo "=============================="

  local cmd=(
    torchrun --standalone --nproc_per_node=8 "nanoGPT/train.py"
    --model_type "${MODEL_TYPE}"
    --tokenizers_path "${TOKENIZERS_PATH}"
    --start_year "${start_year}" --end_year "${end_year}"
    --batch_size "${BATCH_SIZE}"
    --lr "${lr}" --min_lr "${min_lr}" --base_lr "${base_lr}"
    --data_dir "${data_dir}"
    --init_from "${init_from}"
    --record_interval "${RECORD_INTERVAL}"
    --update_interval "${UPDATE_INTERVAL}"
    --num_update "${NUM_UPDATE}"
    --num_epochs "${NUM_EPOCHS}"
    --out_dir "${out_dir}"
    --fix_ratio "${FIX_RATIO}"
    --save_interval "${SAVE_INTERVAL}"
    --warmup_ratio "${WARMUP_RATIO}"
  )

  if [[ -n "${base_dir}" ]]; then
    cmd+=(--base_dir "${base_dir}")
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"

  PREV_OUT_DIR="${out_dir}"
}

for i in "${!DATA_DIRS[@]}"; do
  if [[ "$i" -eq 0 ]]; then
    init_from="scratch"
    base_dir=""
    base_lr="0"
  else
    init_from="finetune"
    base_dir="${PREV_OUT_DIR}"
    base_lr="${CONST_MIN_LR}"
  fi

  # lr = LR_INIT * (LR_DECAY ^ i)
  lr=$(awk -v init="$LR_INIT" -v decay="$LR_DECAY" -v step="$i" 'BEGIN {
    printf "%.12g", init * (decay ^ step)
  }')

  # min_lr: 最后一阶段为 0，其余为 CONST_MIN_LR
  if [[ "$i" -eq $((NUM_STAGES - 1)) ]]; then
    min_lr="0"
  else
    min_lr="${CONST_MIN_LR}"
  fi

  run_stage \
    "${DATA_DIRS[$i]}" \
    "${lr}" \
    "${min_lr}" \
    "${init_from}" \
    "${base_dir}" \
    "${base_lr}"
done