import copy
import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import torch
import torch.distributed as dist
import heapq
from collections import Counter
import os
import pickle

def hash(pair: tuple[int, int]) -> int:
  x, y = pair
  return ((x * 73856093) ^ (y * 19349663)) & 0xFFFFFFFF

class TokenizerChanger:
  # world_size: int                               - Number of processes (Constant)
  # rank: int                                     - DDP rank (Constant)
  # device: str                                   - Device (Constant)
  # space_sign: str                               - Character to replace spaces (Constant)
  # vocab_size: int                               - Fixed size of vocabulary (Constant)
  # model_max_length: int                         - Orignial model_max_length (Constant)
  # original_tokenizer: PreTrainedTokenizerFast   - Original tokenizer (Constant)
  # tokenizer: PreTrainedTokenizerFast            - Tokenizer
  # state: dict                                   - Metadata of tokenizer, used to reconstruct tokenizer
  # vocab: set                                    - Set of vocabulary
  # special_token: list(bool, vocab_size)         - Whether a token is a special token (Constant)
  # merge_set: set                                - Set of pairs of id of merges
  # cnt: tensor(int, vocab_size)                  - Number of composite tokens each token contributes to
  # id2token: list(str, vocab_size)               - Vocabulary with id as the index
  # unused_id: list                               - List of unused id, which length is always less than 1
  # alpha: float                                  - Attenuation Coefficient (Constant)
  # theta: tensor(float, vocab_size)              - Frequency estimator for tokens
  # thetan: tensor(float, vocab_size)             - Lazy tag for delta of token frequency
  # c: tensor(int, vocab_size)                    - Frequency counts of tokens in the latest input chunk
  # candidate_counters: list(Counter, world_size) - Candidate pair frequency counters partitioned by hash value
  # candidate_set: Counter                        - Candidate pair frequency estimator that only includes pairs
  #                                                   whose hash values correspond to the current process rank

  def __init__(self, tokenizer: PreTrainedTokenizerFast = None, space_sign: str = "Ġ", alpha: float = 0.02, tolerance: float = 1.2, device = "cuda", world_size = 1, rank = 0):
    self.alpha = alpha
    self.tolerance = tolerance
    self.world_size = world_size
    self.rank = rank
    self.device = device
    self.space_sign = space_sign
    self.unused_id = []
    
    self.cpu_group = dist.new_group(
      ranks=list(range(world_size)),
      backend='gloo'
    )
    # self.version = 0

    if tokenizer:
      self.tokenizer: PreTrainedTokenizerFast = tokenizer
      self.original_tokenizer = copy.deepcopy(tokenizer)
      self.state = json.loads(
        tokenizer.backend_tokenizer.__getstate__()) if tokenizer else {}
      self.vocab_size = len(
        self.state["model"]["vocab"]) if self.state else 0
      self.vocab = set(self.state["model"]["vocab"].keys())

      self.id2token = [None for _ in range(self.vocab_size)]
      for key in self.state["model"]["vocab"]:
        self.id2token[self.state["model"]["vocab"][key]] = key

      self.cnt = torch.zeros(self.vocab_size).to(device)
      self.special_token = torch.zeros(self.vocab_size).to(device)

      # get cnt
      for add_item in self.state["added_tokens"]:
        self.cnt[add_item["id"]] += 1
        self.special_token[add_item["id"]] = 1
      for token in self.state["model"]["vocab"]:
        if len(token) == 1:
          self.cnt[self.state["model"]["vocab"][token]] += 1 # special token and unit tokens cant be deleted
      
      self.merge_set = []
      for merge in self.state["model"]["merges"]:
        for token in merge:
          self.cnt[self.state["model"]["vocab"][token]] += 1
        self.merge_set.append((self.state["model"]["vocab"][merge[0]], self.state["model"]["vocab"][merge[1]]))
      self.merge_set = set(self.merge_set)
  
      # get theta
      self.theta = torch.zeros(self.vocab_size).to(device)
      self.thetan = torch.zeros(self.vocab_size).to(device)
      self.c = torch.zeros(self.vocab_size).to(device)
      self.candidate_counters = [Counter() for _ in range(world_size)]
      self.candidate_set = Counter()
    
      self.model_max_length = self.tokenizer.model_max_length
      self.updated_tokenizer()


  def delete_merges(self, token: str):
    # print(f"del {token}")
    new_merges = []
    for merge in self.state["model"]["merges"]:
      if "".join(merge) != token:
        new_merges.append(merge)
      else:
        token_id = self.state["model"]["vocab"][token]
        assert self.cnt[token_id] == 0
        self.state["model"]["vocab"].pop(token, None)
        self.vocab.remove(token)
        merge_key = (self.state["model"]["vocab"][merge[0]], self.state["model"]["vocab"][merge[1]])
        self.merge_set.remove(merge_key)
        for sub_token_id in merge_key:
          self.cnt[sub_token_id] -= 1
          self.theta[sub_token_id] += self.thetan[token_id]
          self.thetan[sub_token_id] += self.thetan[token_id]
        assert merge_key not in self.candidate_set
        if hash(merge_key) % self.world_size == self.rank:
          self.candidate_set[merge_key] = float(self.theta[token_id])
        new_candidate_set = {}
        for key in self.candidate_set:
          if key[0] != token_id and key[1] != token_id:
            new_candidate_set[key] = self.candidate_set[key]
        self.candidate_set = Counter(new_candidate_set)
        self.theta[token_id] = 0
        self.thetan[token_id] = 0
        self.unused_id.append(token_id)

    self.state["model"]["merges"] = new_merges
    
    

  def add_merges(self, merge: list[str], init_theta: float=0):
    merge = [token.replace(' ', self.space_sign) for token in merge]
    token = "".join(merge)
    # print(merge, bool(all(token in self.vocab for token in merge)), bool(token not in self.vocab), bool(self.unused_id))
    assert all(token in self.vocab for token in merge) and token not in self.vocab and bool(self.unused_id)

    self.vocab.add(token)
    merge_key = (self.state["model"]["vocab"][merge[0]], self.state["model"]["vocab"][merge[1]])
    self.merge_set.add(merge_key)
    self.state["model"]["merges"].append(merge)
    i = self.unused_id.pop()
    
    for sub_token_id in merge_key:
      self.cnt[sub_token_id] += 1
    self.state["model"]["vocab"][token] = i
    self.id2token[i] = token
    self.theta[i] = init_theta
    self.candidate_set.pop(merge_key, None)
    # if hash(merge_key) % self.world_size == self.rank:
    #   self.candidate_set.pop(merge_key)

  def updated_tokenizer(self):
    """Returns the updated tokenizer

      Returns:
      PreTrainedTokenizerFast: the updated tokenizer
    """

    backend_tokenizer = Tokenizer.from_str(json.dumps(self.state))

    self.tokenizer = self.original_tokenizer.__class__(
      tokenizer_object=backend_tokenizer, **self.original_tokenizer.init_kwargs)

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    self.tokenizer.model_max_length=1e30
    return self.tokenizer
  
  def get_frequency(self):
    if self.world_size != 1:
      dist.all_reduce(self.c, op=dist.ReduceOp.SUM)
      for i in range(self.world_size):
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, self.candidate_counters[i])
        if i == self.rank:
          delta = Counter()
          for part in gathered:
            delta.update(part)
    else:
      delta = self.candidate_counters[0]

    self.theta.mul_(1 - self.alpha).add_(self.c * self.alpha)
    self.thetan.mul_(1 - self.alpha).add_(self.c * self.alpha)
    self.c.zero_()
    
    self.candidate_set = Counter({k: v * (1 - self.alpha) for k, v in self.candidate_set.items()})
    
    delta = Counter({k: v*self.alpha for k, v in delta.items()})
    
    self.candidate_set.update(delta)
  
    # if self.device == 'cuda:0':
      # print(len(self.candidate_list))
    if len(self.candidate_set) > 40000:
      self.candidate_set = Counter(dict(self.candidate_set.most_common(20000)))
    for i in range(self.world_size):
      self.candidate_counters[i].clear()

  def __call__(self, input):
    input_ids = self.tokenizer(input, return_tensors='pt')
    word_ids = torch.tensor(input_ids.word_ids())
    self.count(input_ids['input_ids'][0], word_ids)
    return input_ids

  def _get_pairs(self, input_ids: torch.Tensor, word_ids: torch.Tensor):
      """
      Returns a set of symbol pairs in a word using word IDs.
      
      Args:
          input_ids (torch.Tensor): The input token IDs.
          word_ids (list): A list of word IDs corresponding to each token.
      
      Returns:
          list: A list of (pre_token_id, token_id) pairs.
      """
      word_ids = word_ids.tolist()
      pairs = []
      pre_token_id = -1
      current_word_id = -1
      
      for i, token_id in enumerate(input_ids.tolist()):
          if token_id == self.tokenizer.eos_token_id:
              break
          word_id = word_ids[i]
          
          if word_id != current_word_id:
              pre_token_id = -1
          
          if word_id is None:
              pre_token_id = -1
              current_word_id = word_id
              continue
          
          if pre_token_id != -1:
              pairs.append((pre_token_id, token_id))
          
          pre_token_id = token_id
          current_word_id = word_id
          
      return pairs

  def count(self, input_ids, word_ids_tensor):
    ids = input_ids.clone().detach().to(self.device)
    word_ids = word_ids_tensor.clone().detach().to(self.device)

    # print(ids)

    self.c += torch.bincount(ids, minlength=self.c.size(0))

    # x = ids[:-1]
    # y = ids[1:]
    # mask = (self.special_token[x] == 0) & (self.special_token[y] == 0)

    pairs = self._get_pairs(ids, word_ids) #zip(x[mask].tolist(), y[mask].tolist())
    # print(f"pairs: {pairs}")
    for pair in pairs:
      self.candidate_counters[hash(pair) % self.world_size][pair] += 1
    return
  # First, call get_frequency()
  # Then, call delete_merges() and add_merges() if need
  # Finally, call updated_tokenizer()

  def try_replace(self):
    zero_indices = (self.cnt == 0).nonzero(as_tuple=True)[0]
    min_index = zero_indices[torch.argmin(self.theta[zero_indices])]

    if self.candidate_set:
      max_item = self.candidate_set.most_common(1)[0]
    else:
      max_item = None
    if self.world_size != 1:
      gathered_max = [None for _ in range(self.world_size)]
      dist.all_gather_object(gathered_max, max_item)
      valid_items = [item for item in gathered_max if item is not None]
      if valid_items:
        max_item = max(valid_items, key=lambda x: x[1])
  
    if not max_item:
      return None
    
    if self.theta[min_index] * self.tolerance > max_item[1]:
      return None

    # if self.rank == 0:
    #   print(min_index, self.id2token[min_index], self.theta[min_index])
    #   print(f"({self.id2token[max_item[0][0]]}, {self.id2token[max_item[0][1]]})", max_item[1])
    x, y = max_item[0][0], max_item[0][1]
    
    self.delete_merges(self.id2token[min_index])
    self.add_merges([self.id2token[x], self.id2token[y]], max_item[1])
    return (min_index.item(), x, y)
  
  # Save the tokenizer changer, you must ensure that get_frequency has been called beforehand,
  #   and that the vocabulary size is consistent with the original size.
  def save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None, disable: bool=True):
    assert not self.unused_id

    if self.rank == 0:
      os.makedirs(save_directory, exist_ok=True)
      if self.tokenizer:
        self.tokenizer.model_max_length = self.model_max_length
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"), legacy_format=legacy_format, filename_prefix=filename_prefix)
        self.tokenizer.model_max_length = 1e30
      meta = {
        'alpha': self.alpha,
        'tolerance': self.tolerance,
        'space_sign': self.space_sign,
      }
      with open(os.path.join(save_directory, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
      torch.save(self.theta, os.path.join(save_directory, "theta.pt"))
      torch.save(self.thetan, os.path.join(save_directory, "thetan.pt"))

    if self.world_size != 1:
      dist.barrier()

    with open(os.path.join(save_directory, f"candidate_set_rank{self.rank}.pkl"), "wb") as f:
        pickle.dump(self.candidate_set, f)
    
    if not disable:
      print(f'The tokenizer has been saved to {save_directory}')
  
  def load_pretrained(load_directory, device = "cuda", world_size = 1, rank = 0, disable: bool=True):
    changer = TokenizerChanger()
    meta_path = os.path.join(load_directory, "meta.json")
    if os.path.exists(meta_path):
      with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
      alpha = meta.get("alpha", 0.02)
      toleance = meta.get("tolerance", 1.2)
      space_sign = meta.get("space_sign", "Ġ")
    else:
      print(f"Warning: [Rank {rank}] meta file doesn't find.")
      alpha = 0.02
      toleance = 1.2
      space_sign = "Ġ"
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "tokenizer"))
    changer.__init__(tokenizer, space_sign=space_sign, alpha=alpha, device=device, world_size=world_size, rank=rank)

    changer.theta = torch.load(os.path.join(load_directory, f"theta.pt"), map_location=device)
    changer.thetan = torch.load(os.path.join(load_directory, f"thetan.pt"), map_location=device)

    with open(os.path.join(load_directory, f"candidate_set_rank{rank}.pkl"), "rb") as f:
        changer.candidate_set = pickle.load(f)
    
    if not disable:
      print(f'The tokenizer has been loaded from {load_directory}')
    return changer

if __name__ == "__main__":
  import os
  from torch.distributed import init_process_group, destroy_process_group
  ddp = int(os.environ.get('RANK', -1)) != -1
  if ddp:
    backend = 'nccl'
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
  else:
    # if not ddp, we are running on a single gpu, and one process
    device = 'cuda'
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
  from transformers import GPT2TokenizerFast
  tokenizer = GPT2TokenizerFast.from_pretrained('/data/huangjiameng/checkpoints/openai-community/gpt2/')
  changer = TokenizerChanger(tokenizer, device=device, world_size=ddp_world_size, rank=ddp_rank)
  if master_process:
    print(changer(" 1ABC<SEP>ABC A"))
    print(changer("Compar"))
    print(changer("ComTerry"))
    print(changer("<|endoftext|>"))
    print(changer.cnt[changer.state["model"]["vocab"]["Com"]])
  changer.get_frequency()
  if master_process:
    print(changer.theta[5377], changer.theta[50256])
  # {'input_ids': [5377, 1845], 'attention_mask': [1, 1]}
  # {'input_ids': [5377, 50241], 'attention_mask': [1, 1]}
  # {'input_ids': [5377, 51, 6996], 'attention_mask': [1, 1, 1]}
  # {'input_ids': [50256], 'attention_mask': [1]}
  # tensor(9., device='cuda:0')
  # tensor(0.0200, device='cuda:0') tensor(0.0200, device='cuda:0')

  changer.delete_merges('Compar')
  changer.updated_tokenizer()
  if master_process:
    print(changer("Compar"))
    print(changer("ComTerry"))
    print(changer("<|endoftext|>"))
    print(changer.cnt[changer.state["model"]["vocab"]["Com"]])
  changer.get_frequency()
  if master_process:
    print(changer.theta[5377], changer.theta[50256])
  # {'input_ids': [5377, 1845], 'attention_mask': [1, 1]}
  # {'input_ids': [5377, 50241], 'attention_mask': [1, 1]}
  # {'input_ids': [50256], 'attention_mask': [1]}
  # tensor(8., device='cuda:0')
  # tensor(0.0792, device='cuda:0') tensor(0.0396, device='cuda:0')

  changer.add_merges(["Com","Terry"])
  changer.updated_tokenizer()
  if master_process:
    print(changer("Compar"))
    print(changer("ComTerry"))
    print(changer("<|endoftext|>"))
    print(changer.cnt[changer.state["model"]["vocab"]["Com"]])
  changer.get_frequency()
  if master_process:
    print(changer.theta[5377], changer.theta[50256])
  # tensor(0.0976, device='cuda:0') tensor(0.0588, device='cuda:0')
  # {'input_ids': [5377, 1845], 'attention_mask': [1, 1]}
  # {'input_ids': [50249], 'attention_mask': [1]}
  # {'input_ids': [50256], 'attention_mask': [1]}
  # tensor(9., device='cuda:0')
  # tensor(0.0976, device='cuda:0') tensor(0.0588, device='cuda:0')
  
  print(f'[Rank {ddp_rank}]: candidate_set: {changer.candidate_set}')

  changer.save_pretrained('/data/zhangzhi/temp/tokenizer_test', disable=False)
  changer2 = TokenizerChanger.load_pretrained('/data/zhangzhi/temp/tokenizer_test', device=device, world_size=ddp_world_size, rank=ddp_rank, disable=False)
  # The tokenizer has been saved to /data/zhangzhi/temp/tokenizer_test
  # The tokenizer has been loaded from /data/zhangzhi/temp/tokenizer_test

  print(f'[Rank {ddp_rank}]: candidate_set: {changer.candidate_set}')
  if master_process:
    print(changer2.theta[5377], changer2.theta[50256])
  if master_process:
    print(changer2("Compar"))
    print(changer2("ComTerry"))
    print(changer2("<|endoftext|>"))
    print(changer2.cnt[changer.state["model"]["vocab"]["Com"]])
  changer2.get_frequency()
  if master_process:
    print(changer2.theta[5377], changer2.theta[50256])
  # {'input_ids': [5377, 1845], 'attention_mask': [1, 1]}
  # {'input_ids': [50249], 'attention_mask': [1]}
  # {'input_ids': [50256], 'attention_mask': [1]}
  # tensor(9., device='cuda:0')
  # tensor(0.1157, device='cuda:0') tensor(0.0776, device='cuda:0')

  print(f'[Rank {ddp_rank}]: candidate_set: {changer2.candidate_set}')
  # for i in range(50):
  #   changer.get_frequency()
  #   changer.delete_merges('ComTerry')
  #   changer.add_merges(["Com","par"])
  #   changer.updated_tokenizer()

  #   changer.get_frequency()
  #   changer.delete_merges('Compar')
  #   changer.add_merges(["Com","Terry"])
  #   changer.updated_tokenizer()
  if ddp:
    destroy_process_group()