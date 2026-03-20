#!/bin/bash
set -e

DATA_PATH="./data/WMT_rawtext/wmt_2019-2021"
PREV_OUT_DIR="./checkpoints/models/ours-small-10epoch_finetune_wmt_2016-2018_tok2007-2009"
TOKENIZERS_PATH="./checkpoints/tokenizers"
TOK_START_YEAR=2019
TOK_END_YEAR=2021
OUTPUT_PATH="./checkpoints/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --prev_out_dir)
      PREV_OUT_DIR="$2"
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
    --tok_start_year)
      TOK_START_YEAR="$2"
      shift 2
      ;;
    --tok_end_year)
      TOK_END_YEAR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--data_path PATH] [--prev_out_dir DIR] [--tokenizers_path PATH] [--output_path PATH] [--tok_start_year YEAR] [--tok_end_year YEAR]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

if [ ! -d "${PREV_OUT_DIR}" ]; then
  echo "Error: 找不到上一阶段的目录: ${PREV_OUT_DIR}"
  echo "请检查该目录是否存在，或者是否在正确的根目录下运行脚本。"
  exit 1
fi

MODEL_TYPE=small

BATCH_SIZE=36
SAVE_INTERVAL=2000
UPDATE_INTERVAL=-1
RECORD_INTERVAL=-1
NUM_UPDATE=100
FIX_RATIO=0.7
TOK_TO_COMPARE=${TOKENIZERS_PATH}/custom_gpt2_tokenizer_2019-2021

LR_1921=0.00031320375 
MINLR_1921=0          
BASELR_1921=0.00006    
NUM_EPOCHS=10
WARMUP_RATIO=0.04

mkdir -p logs

# 一个小函数，负责跑单个阶段
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

run_stage "${DATA_PATH}" "${LR_1921}" "${MINLR_1921}" "finetune" "${PREV_OUT_DIR}" "${BASELR_1921}" 
