#!/bin/bash

# 默认值
DATA_PATH="./data/WMT"
OUTPUT_RAWTEXT_PATH="./data/WMT_rawtext"
OUTPUT_TOKENIZER_PATH="./checkpoints/tokenizers"
START_YEAR=2019
END_YEAR=2021

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_raw_path)
      OUTPUT_RAWTEXT_PATH="$2"
      shift 2
      ;;
    --output_tokenizer_path)
      OUTPUT_TOKENIZER_PATH="$2"
      shift 2
      ;;
    --start_year)
      START_YEAR="$2"
      shift 2
      ;;
    --end_year)
      END_YEAR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--data_path PATH] [--output_rawtext_path PATH] [--output_tokenizer_path PATH] [--start_year YEAR] [--end_year YEAR]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "DATA_PATH: $DATA_PATH"
echo "OUTPUT_RAWTEXT_PATH: $OUTPUT_RAWTEXT_PATH"
echo "OUTPUT_TOKENIZER_PATH: $OUTPUT_TOKENIZER_PATH"
echo "START_YEAR: $START_YEAR"
echo "END_YEAR: $END_YEAR"

if [ ! -f "$OUTPUT_RAWTEXT_PATH/wmt_${START_YEAR}-${END_YEAR}.raw" ]; then
  echo "Raw text file not found, starting extraction..."
  python data_processer/extraction_by_years.py --start_year $START_YEAR --end_year $END_YEAR --data_path $DATA_PATH --output_path $OUTPUT_RAWTEXT_PATH
fi

python ../tokenizer_module/build_tokenizer.py --input_file "$OUTPUT_RAWTEXT_PATH/wmt_${START_YEAR}-${END_YEAR}.raw" --save_dir "$OUTPUT_TOKENIZER_PATH/custom_gpt2_tokenizer_${START_YEAR}-${END_YEAR}"

echo "Tokenizer building completed!"