#!/bin/bash

# 默认值
DATA_PATH="./data/WMT"
OUTPUT_PATH="./data/WMT_rawtext"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--data_path PATH] [--output_path PATH]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "DATA_PATH: $DATA_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"

START=2007
END=2021
STEP=3

current_start=$START

while [ $current_start -le $END ]; do
  current_end=$((current_start + STEP - 1))

  # 防止超过总结束年份
  if [ $current_end -gt $END ]; then
    current_end=$END
  fi

  echo "Processing: ${current_start}-${current_end}"

  python data_processer/extraction_by_years.py \
    --start_year "$current_start" \
    --end_year "$current_end" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH"

  current_start=$((current_start + STEP))
done