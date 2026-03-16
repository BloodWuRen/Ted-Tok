import os
import json
import argparse

from tokenizers import (
    Tokenizer,
    pre_tokenizers,
    models,
    decoders,
    trainers,
    processors,
)
from transformers import GPT2TokenizerFast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WMT pretraining data.")
    parser.add_argument("--start_year", type=int, default=2007,
                        help="Start year for document extraction (inclusive)")
    parser.add_argument("--end_year", type=int, default=2007,
                        help="End year for document extraction (inclusive)")
    args = parser.parse_args()

    raw_text_path = f'/data/zhangzhi/data/local_datasets/wmt_{args.start_year}-{args.end_year}.raw'
    VOCAB_SIZE = 50257  # GPT-2 默认词汇表大小

    files = [raw_text_path]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()

    trainer = trainers.BpeTrainer(
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        vocab_size=VOCAB_SIZE,
        special_tokens=[
            "<pad>", "<|beginoftext|>", "<|endoftext|>"
        ]
    )

    tokenizer.train(files, trainer)
    test_text = "Hello World!"

    print("pre-tokenize spans:", tokenizer.pre_tokenizer.pre_tokenize_str(test_text))
    ids = tokenizer.encode(test_text).ids
    print(f"tokens: {[tokenizer.decode([tid]) for tid in ids]}")

    save_dir = f"/data/zhangzhi/streamingllm_pre_experiment/nanoGPT/custom_gpt2_tokenizer_{args.start_year}-{args.end_year}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.save_pretrained(save_dir)