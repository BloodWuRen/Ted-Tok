from collections import defaultdict
import os
from model import TransformerLanguageModel
import torch
import re
import random
import numpy as np
from tqdm import tqdm
import json
import copy
from torch.utils.tensorboard import SummaryWriter

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer LM with different tokenizers.")
    parser.add_argument(
        "--train_type",
        type=str,
        default="C",
        choices=["C", "D", "E"],
        help="Tokenizer type: C (old cpt), D (ted cpt), E(new cpt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g., 'cuda:0' or 'cpu'). If not specified, will use 'cuda:0' if available, otherwise 'cpu'."
    )
    parser.add_argument(
        "--old_dataset_path",
        type=str,
        default="data/arith_lt50.txt",
        help="Path to the old dataset."
    )
    parser.add_argument(
        "--new_dataset_path",
        type=str,
        default="data/arith_lt200_train.txt",
        help="Path to the new dataset."
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/arith_lt200_test.txt",
        help="Path to the test dataset (used for evaluation)."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/",
        help="Path to save model checkpoints."
    )
    return parser.parse_args()

def set_seed(seed: int = 19, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    else:
        torch.backends.cudnn.benchmark = True

from typing import Iterable

def jaccard_distance(a: Iterable[str], b: Iterable[str]) -> float:
    """
    计算两个字符串数组的 Jaccard 距离：1 - |A∩B| / |A∪B|
    - 输入会先转为 set 以去重
    - 若并集为空（两边都空），返回 0.0
    """
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    inter = sa & sb
    print(len(inter))
    print(len(union))
    return 1.0 - len(inter) / len(union)

def build_trie(vocab):
    trie_node = {}
    for token in vocab:
        node = trie_node
        for char in token:
            if char not in node:
                node[char] = {}
            node = node[char]
    return trie_node

def run_bpe_build_vocab(text_segments, vocab_limit, num_iterations):
    # 提取所有初始字符
    print(text_segments[:20])
    initial_chars = sorted(list(set(c for segment in text_segments for c in segment if c.isdigit() or c in '+-^=')))
    active_vocabulary = set(initial_chars)
    merges = []

    # BPE 迭代循环
    for _ in range(num_iterations):
        current_trie = build_trie(active_vocabulary)
        pair_counts = defaultdict(int)
        
        # 遍历每个文本段进行分词和统计词对
        for segment in text_segments:
            if not segment: # 跳过空段
                continue
            
            tokenized_segment = []
            i = 0
            while i < len(segment):
                current_match_prefix = ""
                longest_match = ""
                match_end_index = i
                temp_node = current_trie
                k = i

                # 在Trie中查找最长匹配
                while k < len(segment) and segment[k] in temp_node:
                    temp_node = temp_node[segment[k]]
                    current_match_prefix += segment[k]
                    if current_match_prefix in active_vocabulary:
                        longest_match = current_match_prefix
                        match_end_index = k + 1
                    k += 1

                # 如果没有匹配，则取当前字符作为最长匹配
                if not longest_match:
                    longest_match = segment[i]
                    match_end_index = i + 1

                tokenized_segment.append(longest_match)

                i = match_end_index
            
            # 统计词对频率
            for j in range(len(tokenized_segment) - 1):
                pair_counts[(tokenized_segment[j], tokenized_segment[j+1])] += 1

        if not pair_counts:
            # print("\n没有更多词对可以合并。提前退出。")
            break

        merge_pair = max(pair_counts, key=pair_counts.get)
        new_token = "".join(merge_pair)

        if new_token not in active_vocabulary:
            active_vocabulary.add(new_token)
            merges.append(merge_pair)
        if len(active_vocabulary) >= vocab_limit:
            break
    
    return active_vocabulary, merges

def tokenize_expression(expression, vocabulary, trie):
    """
    使用构建好的词汇表和Trie对新的表达式进行分词。
    """
    tokenized_result = []
    i = 0
    processed_expression = "".join(c for c in expression if c.isdigit() or c in '+-^=')

    while i < len(processed_expression):
        current_match_prefix = ""
        longest_match = ""
        match_end_index = i
        temp_node = trie
        k = i

        while k < len(processed_expression) and processed_expression[k] in temp_node:
            temp_node = temp_node[processed_expression[k]]
            current_match_prefix += processed_expression[k] 
            if current_match_prefix in vocabulary:
                longest_match = current_match_prefix
                match_end_index = k + 1
            k += 1

        if not longest_match:
            if i < len(processed_expression):
                longest_match = processed_expression[i]
                match_end_index = i + 1
            else:
                break

        tokenized_result.append(longest_match)
        i = match_end_index
    return tokenized_result

def BPE_tokenizer_build(data_raw, vocab_limit=50, iterations=50):
    pre_tokenized_segments = []
    for segment in data_raw:
        if segment.strip():
            pre_tokenized_segments += re.findall(r'[\d+\^\-]+|[=]',segment.strip())

    print(f"### 预分词后的文本段数量: {len(pre_tokenized_segments)}")

    vocab, merges = run_bpe_build_vocab(
        pre_tokenized_segments,
        vocab_limit,
        iterations
    )
    return vocab, merges

args = parse_args()
set_seed()

print(f"\n\n{'#'*60}")
print(f"### 语言模型训练与预测 (CUDA Enabled) ###")
print(f"{'#'*60}\n")

if torch.cuda.is_available():
    target_device = args.device if args.device is not None else "cuda:0"
device = torch.device(target_device) if torch.cuda.is_available() else torch.device("cpu")
print(f"模型将在 {device} 上训练。")

with open(args.old_dataset_path, 'r') as file:
    data_raw_small = file.read().strip().split('\n')
with open(args.new_dataset_path, 'r') as file:
    data_raw_big = file.read().strip().split('\n')
with open(args.test_dataset_path, 'r') as file:
    test_data_raw = file.read().strip().split('\n')
print(f"test dataset size: {len(test_data_raw)}")
print(f"train dataset size: {len(data_raw_big)}")


train_type = args.train_type
final_vocabulary, merges = BPE_tokenizer_build(data_raw_small, vocab_limit=50, iterations=50)
if train_type == 'D':
    from tokenizers import Tokenizer, models
    from transformers import GPT2TokenizerFast
    from ..tokenizer_module.TokenizerChanger import TokenizerChanger

    special_tokens = {"<S>", "\n", "<UNK>"}
    init_vocabulary = sorted(list(final_vocabulary.union(special_tokens)))

    vocab_dict = {token: idx for idx, token in enumerate(init_vocabulary)}
    tokenizer_obj = Tokenizer(models.BPE(
        vocab=vocab_dict,
        merges=merges,
        dropout=None,
        unk_token=None
    ))
    tokenizer = GPT2TokenizerFast(
        tokenizer_object=tokenizer_obj,
        eos_token="\n",
        bos_token="<S>",
        unk_token="<UNK>",
        pad_token="=",
        model_max_length=1024,
    )

    changer = TokenizerChanger(tokenizer, alpha=0.3, device=device)
new_vocabulary, _ = BPE_tokenizer_build(data_raw_big, vocab_limit=50, iterations=50)

print(f"\n\n{'#'*60}")
print(f"### 词汇表构建完成 ###")
print(f"最终活跃词汇表 ({len(final_vocabulary)} 个词元): {sorted(list(final_vocabulary))}")
print(f"{'#'*60}\n")

lm = TransformerLanguageModel(final_vocabulary, device=device)
print(lm.get_num_parameters())
if train_type == 'D':
    assert all(lm.id_to_token[i] == changer.id2token[i] for i in range(len(changer.id2token)))

writer = SummaryWriter(log_dir=f'runs/{train_type}-{lm.lr}')
def train_lm_on_chunk(lm, final_vocabulary, offset, epoch_offset, lm_training_data_raw, test_data_raw, chunk_num, epoch_num, revise_enabled=False):
    for i in range(chunk_num):
        lm_tokenized_sequences = []
        lm_val_sequences = []
        tmp = []
        final_trie = build_trie(final_vocabulary)
        for line in lm_training_data_raw:
            if line.strip():
                tokenized_line = tokenize_expression(line, final_vocabulary, final_trie)
                lm_tokenized_sequences.append(tokenized_line)
                tmp.append(len(tokenized_line))
        for line in test_data_raw:
            if line.strip():
                tokenized_line = tokenize_expression(line, final_vocabulary, final_trie)
                lm_val_sequences.append(tokenized_line)
        print(f"平均编码长度: {sum(tmp) / len(tmp)}")
        print(final_vocabulary)
        writer.add_scalar('AL', sum(tmp) / len(tmp), i)
        writer.add_scalar('dist', jaccard_distance(new_vocabulary, final_vocabulary), i)

        (offset, epoch_offset) = lm.fit(lm_tokenized_sequences, lm_val_sequences, epochs=epoch_num // chunk_num, offset=offset, epoch_offset=epoch_offset, writer=writer)
        
        print(f'Chunk {i} finish.')
        if i < chunk_num - 10 and revise_enabled:
            dataset_to_count = copy.deepcopy(lm_training_data_raw)
            random.shuffle(dataset_to_count)
            dataset_to_count = dataset_to_count[:4193]
            for j in tqdm(range(len(dataset_to_count))):
                for segment in dataset_to_count[j].split('='):
                    changer(segment)
                if (j + 1) % 599 == 0:
                    changer.get_frequency()
            if i < 4:
                continue
            _ = changer.try_replace()
            if _ is not None:
                id, x, y = _
                x = lm.token_to_id[changer.id2token[x]]
                y = lm.token_to_id[changer.id2token[y]]
                changer.updated_tokenizer()
                new_token = changer.id2token[id]

                lm.token_to_id[new_token] = id
                lm.id_to_token[id] = new_token
                assert all(lm.id_to_token[i] == changer.id2token[i] for i in range(len(changer.id2token)))

                W = lm.embedding.weight
                O = lm.output_linear.weight
                with torch.no_grad():
                    new_W = (W[x] + W[y]) / 2.0
                    W[id].copy_(new_W)

                    new_O = (O[x] + O[y]) / 2.0
                    O[id].copy_(new_O)
                final_vocabulary = changer.vocab
                lm.set_vocabulary(final_vocabulary)
    return lm, final_vocabulary, offset, epoch_offset

offset = 0
epoch_offset = 0

epoch_num = 1000


lm, final_vocabulary, offset, epoch_offset = train_lm_on_chunk(lm, final_vocabulary, offset, epoch_offset, data_raw_small, test_data_raw, 1, epoch_num, False)
if train_type == 'D':
    chunk_num = 50
else:
    chunk_num = 1
if train_type == 'E':
    final_vocabulary, _ = BPE_tokenizer_build(data_raw_big, vocab_limit=50, iterations=50)
    lm.reset_vocabulary(final_vocabulary)
lm, final_vocabulary, offset, epoch_offset = train_lm_on_chunk(lm, final_vocabulary, offset, epoch_offset, data_raw_big, test_data_raw, chunk_num, epoch_num, train_type == 'D')

writer.close()
print("\n语言模型训练完成。")

# exit()

def generate_sequence(model, initial_context_token, max_gen_length=15):
    # print(f"\n起始上下文: '{initial_context_token}'")
    trie = build_trie(set(model.vocabulary))
    temp_tokenized = tokenize_expression(initial_context_token, model.vocabulary, trie)
    predicted_sequence = list(temp_tokenized)

    full_generated_text = ""
    for _ in range(max_gen_length):
        next_token = model.predict_next_token(predicted_sequence) 
        if next_token == model.end_token:
            break
        predicted_sequence.append(next_token)
        full_generated_text += next_token
    return full_generated_text

os.makedirs(args.checkpoint_path, exist_ok=True)
torch.save(lm.state_dict(), f"{args.checkpoint_path}/model_weights_{train_type}.pth")

ac = 0
for line in data_raw_big[:100]:
  head, sep, tail = line.rpartition("=")
  if generate_sequence(lm, head + sep) == tail:
    ac += 1
print(f"acc on train dataset : {ac}%")

ac = 0
for line in test_data_raw:
  head, sep, tail = line.rpartition("=")
  if generate_sequence(lm, head + sep) == tail:
    ac += 1
print(f"acc on val dataset: {ac*100.0/len(test_data_raw)}%")