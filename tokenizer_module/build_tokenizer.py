import os
import json
import argparse
from typing import Iterable, List

from tokenizers import (
    Tokenizer,
    pre_tokenizers,
    models,
    decoders,
    trainers,
    processors,
)
from transformers import GPT2TokenizerFast

def infer_input_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    format_map = {
        ".raw": "text",
        ".txt": "text",
        ".text": "text",
        ".parquet": "parquet",
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".json": "jsonl",
    }
    if ext not in format_map:
        raise ValueError(
            f"Cannot infer input format from extension '{ext}'. "
            "Please set --input_format explicitly."
        )
    return format_map[ext]

def clean_text(text: str) -> str:
    return text.strip().replace("<SEP>", "")

def iter_text_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = clean_text(line)
            if text:
                yield text

def select_text_columns(df) -> List[str]:
    inferred_columns = [
        col for col in df.columns
        if str(df[col].dtype) in ("object", "string")
    ]
    if inferred_columns:
        return inferred_columns

    return list(df.columns)

def iter_row_texts(row, columns) -> Iterable[str]:
    for col in columns:
        value = row[col]
        if isinstance(value, str):
            text = clean_text(value)
            if text:
                yield text

def iter_tabular_rows(path: str, input_format: str) -> Iterable[str]:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Reading parquet/csv/jsonl requires pandas. "
            "For parquet, you may also need pyarrow or fastparquet."
        ) from e

    if input_format == "parquet":
        df = pd.read_parquet(path)
    elif input_format == "csv":
        df = pd.read_csv(path)
    elif input_format == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported tabular input format: {input_format}")

    columns = select_text_columns(df)
    print(f"Selected columns for tokenizer training: {columns}")
    for _, row in df.iterrows():
        for text in iter_row_texts(row, columns):
            yield text

def build_training_corpus(path: str, input_format: str) -> Iterable[str]:
    if input_format == "text":
        return iter_text_lines(path)
    if input_format in {"parquet", "csv", "jsonl"}:
        return iter_tabular_rows(path, input_format)
    raise ValueError(f"Unsupported input format: {input_format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from text or tabular data.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path.",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="auto",
        choices=["auto", "text", "parquet", "csv", "jsonl"],
        help="Input file format. Default: infer from file extension.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='checkpoints/custom_gpt2_tokenizer_2007-2007',
        help="Path to save the trained tokenizer"
    )
    parser.add_argument(
        "--SEP",
        action="store_true",
        help="Whether to add special token <SEP>."
    )
    args = parser.parse_args()

    input_format = infer_input_format(args.input_file) if args.input_format == "auto" else args.input_format
    train_corpus = build_training_corpus(args.input_file, input_format)

    VOCAB_SIZE = 50257  # GPT-2 默认词汇表大小
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()

    special_tokens = ["<pad>", "<|beginoftext|>", "<|endoftext|>"]
    if args.SEP:
        special_tokens.append("<SEP>")

    trainer = trainers.BpeTrainer(
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(train_corpus, trainer)
    test_text = "Hello World!"

    print("pre-tokenize spans:", tokenizer.pre_tokenizer.pre_tokenize_str(test_text))
    ids = tokenizer.encode(test_text).ids
    print(f"tokens: {[tokenizer.decode([tid]) for tid in ids]}")

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.save_pretrained(save_dir)