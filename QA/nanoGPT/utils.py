import os
import re
import math

import torch
from torch.distributed import broadcast


def find_latest_checkpoint(out_dir):
    latest_iter = -1
    latest_ckpt_path = None
    pattern = re.compile(r'iter(\d+)_ckpt\.pt')
    print(f"Searching for latest checkpoint in {out_dir}...")
    for filename in os.listdir(out_dir):
        match = pattern.match(filename)
        if match:
            iter_num = int(match.group(1))
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest_ckpt_path = os.path.join(out_dir, filename)
    print(f"Latest checkpoint: {latest_ckpt_path} at iteration {latest_iter}")
    return latest_ckpt_path

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, lr_decay_iters, warmup_iters, learning_rate, min_lr, base_rate):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return (learning_rate - base_rate) * (it + 1) / (warmup_iters + 1) + base_rate
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def get_batch(split: str, split_dataset, changer, block_size, batch_size, update_interval, device):
    ds = split_dataset[split]
    # global batch_size
    sample_factor = batch_size // 3 # empirical value for better token utilization
    idx = torch.randint(0, len(ds), (batch_size * sample_factor,), device='cpu').tolist()
    texts = [ds[i]['text'] for i in idx]

    eos_id = changer.tokenizer.eos_token_id

    enc = changer.tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False
    )

    all_ids = []
    for b, ids in enumerate(enc["input_ids"]):
        if update_interval > 0:
            changer.count(torch.tensor(ids), torch.tensor(enc.word_ids(b)))
        all_ids.extend(ids + [eos_id])

    all_ids = torch.tensor(all_ids, dtype=torch.long)
    total_len = len(all_ids)
    local_block_size = min(total_len - 1, block_size)

    need = batch_size * local_block_size + 1
    if total_len >= need:
        # 长度够就切成不相交切片：起点 0, L, 2L, ...
        starts = torch.arange(0, batch_size * local_block_size, step=local_block_size)
        x = torch.stack([all_ids[s : s + local_block_size] for s in starts])
        y = torch.stack([all_ids[s + 1 : s + 1 + local_block_size] for s in starts])
    else:
        # 长度不够时回退到随机采片段
        max_start = max(1, total_len - local_block_size)
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([all_ids[i : i + local_block_size] for i in ix])
        y = torch.stack([all_ids[i + 1 : i + 1 + local_block_size] for i in ix])

    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def evaluate(model, split_dataset, changer, ctx, batch_size, eval_iters, iter_num, update_interval, device, writer):
    global val_loss

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split = split,
                                    split_dataset = split_dataset, 
                                    changer = changer, 
                                    batch_size = batch_size,
                                    update_interval = update_interval,
                                    block_size = model.module.config.block_size,
                                    device = device,
                                 ) # block_size = model.config.block_size
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    losses = estimate_loss()
    val_loss = losses['val']
    writer.add_scalar('Validation Loss', float(val_loss), iter_num)
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}")


def save_checkpoint(master_process, iter_num, out_dir, raw_model, optimizer, model_args, changer):
    if master_process:
        if iter_num > 0: # 避免在初始时保存
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'iter{iter_num}_ckpt.pt'))
    changer.save_pretrained(os.path.join(out_dir, f"changer_iter{iter_num}"))


# def broadcast_bool_from_rank0(local_flag: bool, device) -> bool:
#     t = torch.tensor([int(local_flag)], device=device)
#     broadcast(t, src=0)
#     return bool(t.item())

def calc_jaccard_distance(tok_old, tok_new):
    vocab_old = set(tok_old.get_vocab().keys())
    vocab_new = set(tok_new.get_vocab().keys())

    intersection = len(vocab_old & vocab_new)
    union = len(vocab_old | vocab_new)
    jaccard_similarity = intersection / union
    jaccard_distance = 1 - jaccard_similarity

    return jaccard_distance