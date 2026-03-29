import os
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
import argparse

def initialize_and_save_model_from_scratch(tokenizer_path: str, save_path: str):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  config = GPT2Config.from_pretrained("/data/huangjiameng/checkpoints/openai-community/gpt2")
  model = GPT2LMHeadModel(config)
  os.makedirs(save_path, exist_ok=True)
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Initialize and save a GPT-2 model from scratch using a custom tokenizer.")
  parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the custom tokenizer directory.")
  parser.add_argument("--save_path", type=str, required=True, help="Path to save the initialized model and tokenizer.")
  args = parser.parse_args()

  custom_tokenizer_dir = args.tokenizer_path
  output_dir = args.save_path
  initialize_and_save_model_from_scratch(custom_tokenizer_dir, output_dir)