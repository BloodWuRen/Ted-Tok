/data/huangjiameng/temp/en-cs_wmt_tokenizers/custom_tokenizer_wmt19

INPUT_DIR=data/wmt09/de-en/wmt09_en_de_decoder_only_dataset/train/train.parquet
SAVE_DIR=checkpoints/tokenizers/custom_tokenizer_wmt19

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--input_dir PATH] [--save_dir PATH]"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "INPUT_DIR: $INPUT_DIR"
echo "SAVE_DIR: $SAVE_DIR"

python ../tokenizer_module/build_tokenizer.py --input_file $INPUT_DIR --save_dir $SAVE_DIR