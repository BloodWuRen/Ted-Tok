"""This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import argparse
from contextlib import nullcontext
from datasets import load_from_disk
import os
import time
import pickle

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.utils.tensorboard import SummaryWriter # 导入 SummaryWriter
from transformers import GPT2TokenizerFast

from model import GPTConfig, GPT
from TokenizerChanger import TokenizerChanger
from utils import find_latest_checkpoint, get_lr, get_batch, evaluate, save_checkpoint, calc_jaccard_distance

parser = argparse.ArgumentParser(description="Process WMT pretraining data.")
parser.add_argument("--tokenizers_path", type=str, default='tokenizers',
                    help="Path to the tokenizers")
parser.add_argument("--start_year", type=int, default=2007,
                    help="Start year for document extraction (inclusive)")
parser.add_argument("--end_year", type=int, default=2007,
                    help="End year for document extraction (inclusive)")
parser.add_argument("--data_dir", type=str, default='wmt_2009-2021',
                    help="Directory of the dataset")
parser.add_argument("--base_dir", type=str, default='ours-nano-customtok-3epoch_scratch_wmt_2007-2012_tok2007-2009',
                    help="Directory of the base model (if init_from == 'finetune')")
parser.add_argument("--out_dir", type=str, default='outdir',
                    help="Output directory for checkpoints and logs")
parser.add_argument("--init_from", type=str, default='scratch',
                    help="'finetune' or 'scratch' or 'resume' or 'gpt2*'")
parser.add_argument("--dtype", type=str, default='float32',
                    help="'float32', 'bfloat16' or 'float16'")

parser.add_argument("--num_epochs", type=int, default=10,
                    help="Number of epochs to train.")
parser.add_argument("--lr_decay_epochs", type=int, default=-1,
                    help="Number of epochs over which to decay LR. -1 to set default as num_epochs.")
parser.add_argument("--batch_size", type=int, default=36,
                    help="Interval for saving model checkpoints.")
parser.add_argument("--lr", type=float, default=6e-4,
                    help="max learning rate.")
parser.add_argument("--base_lr", type=float, default=0,
                    help="starting learning rate.")
parser.add_argument("--min_lr", type=float, default=0,
                    help="min learning rate.")
parser.add_argument("--eval_interval", type=int, default=2000,
                    help="Interval for evaluation.")
parser.add_argument("--save_interval", type=int, default=2000,
                    help="Interval for saving model checkpoints.")
parser.add_argument("--warmup_ratio", type=float, default=0.04,
                    help="ratio of warmup steps.")

parser.add_argument("--alpha", type=float, default=0.00005,
                    help="EMA smooth facter.")
parser.add_argument("--fix_ratio", type=float, default=0.7,
                    help="Fix tokenizer updating after this ratio of total iters.")
parser.add_argument("--record_interval", type=int, default=-1,
                    help="Interval for recording token frequencies. -1 to disable.")
parser.add_argument("--update_interval", type=int, default=-1,
                    help="Interval for updating the tokenizer. -1 to disable.")
parser.add_argument("--num_update", type=int, default=1,
                    help="How many token to update each turn.")
parser.add_argument("--model_type", type=str, default='small',
                    help="'small' or 'medium'")
parser.add_argument("--tok_to_comp", type=str, default='',
                    help="The tokenizer to compare")


args = parser.parse_args()
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

num_epochs = args.num_epochs
data_dir = args.data_dir
out_dir = args.out_dir
# out_dir = f"ours-{args.model_type}-{num_epochs}epoch_{args.init_from}_{os.path.basename(data_dir)}_tok{args.start_year}-{args.end_year}{'' if args.update_interval <= 0 else f'_step{args.update_interval}'}"
base_dir = args.base_dir
tokenizer_dir=os.path.join(args.tokenizers_path, f"custom_gpt2_tokenizer_{args.start_year}-{args.end_year}")
tok_to_compare = GPT2TokenizerFast.from_pretrained(args.tok_to_comp) if args.tok_to_comp != '' else None


# if args.model_type == 'small':
learning_rate = args.lr # max learning rate
# elif args.model_type == 'medium':
#     learning_rate = 3e-4 # max learning rate
# else:
#     raise ValueError(f"Unknown model_type: {args.model_type}")
base_rate = args.base_lr # 0 # learning rate at the start of training
min_lr = args.min_lr # 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
batch_size = args.batch_size

eval_interval = args.eval_interval
save_interval = args.save_interval
log_interval = 1
dtype = args.dtype 

# Streaming Settings
record_interval = args.record_interval
update_interval = args.update_interval


# =========== model/data/hyperparameters that are usually fixed BEGIN ==============
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = args.init_from # 'scratch' or 'resume' or 'gpt2*' or 'finetune'
# data
num_proc = 8
gradient_accumulation_steps = (72 // batch_size) * 8 # used to simulate larger batch sizes
# model
if args.model_type == 'small':
    n_layer = 12
    n_head = 12
    n_embd = 768
elif args.model_type == 'medium':
    n_layer = 24
    n_head = 16
    n_embd = 1024
else:
    raise ValueError(f"Unknown model_type: {args.model_type}")
block_size = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' 
compile = True 
# =========== model/data/hyperparameters that are usually fixed END ==============

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

import time
time_offset = int(time.time()) % 100000
torch.manual_seed(time_offset + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# data loading
dataset = load_from_disk(data_dir)
split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')
dataset_len = len(split_dataset['train'])
lr_decay_epochs = args.lr_decay_epochs if args.lr_decay_epochs > 0 else num_epochs
lr_decay_iters = dataset_len * lr_decay_epochs // (ddp_world_size * batch_size * gradient_accumulation_steps)
warmup_iters = args.warmup_ratio * lr_decay_iters # how many steps to warm up for
fix_iters = int(args.fix_ratio * (dataset_len * num_epochs // (ddp_world_size * batch_size * gradient_accumulation_steps)))
print("lr_decay_iters:", lr_decay_iters)
print("warmup_iters:", warmup_iters)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
changer = TokenizerChanger(tokenizer, alpha=args.alpha, world_size=ddp_world_size, rank=ddp_rank)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = find_latest_checkpoint(out_dir)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    changer = TokenizerChanger.load_pretrained(os.path.join(out_dir, f"changer_iter{iter_num}"), world_size=ddp_world_size, rank=ddp_rank)

elif init_from == 'finetune':
    print(f"Finetuning based on {base_dir}")
    # finetune based on a checkpoint.
    ckpt_path = find_latest_checkpoint(base_dir)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    changer = TokenizerChanger.load_pretrained(os.path.join(base_dir, f"changer_iter{iter_num}"), world_size=ddp_world_size, rank=ddp_rank)
    iter_num = 0

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# # optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# --- TensorBoard Setup ---
log_dir = f"runs/exp_wmt_train_{args.model_type}-{args.init_from}_{data_dir}_tok{args.start_year}-{args.end_year}{'' if args.update_interval <= 0 else f'_step{args.update_interval}'}_{time.strftime('%Y%m%d-%H%M%S')}"
if master_process:
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs are being saved to: {log_dir}")
# --- End TensorBoard Setup ---

local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = None
if dtype == 'float16':
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

model.train()
X, Y = get_batch(split = 'train', 
                 split_dataset = split_dataset, 
                 changer = changer, 
                 batch_size = batch_size,
                 update_interval = update_interval,
                 block_size = block_size,
                 device=device,
                 )  # fetch the very first batch

local_iter_num = 0
max_iters = dataset_len * num_epochs // (ddp_world_size * batch_size * gradient_accumulation_steps)
t0 = time.time()
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(it = iter_num, 
                lr_decay_iters = lr_decay_iters, 
                warmup_iters = warmup_iters, 
                learning_rate = learning_rate, 
                min_lr = min_lr, 
                base_rate = base_rate
                ) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # --- TensorBoard: Log learning rate ---
    if master_process:
        writer.add_scalar('Learning Rate', lr, iter_num)
    # --- End TensorBoard ---

    accum_loss_cpu = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps    

        accum_loss_cpu += float(loss.detach())
        if dtype == 'float16':
            scaler.scale(loss).backward()
        elif dtype == 'bfloat16' or dtype == 'float32':
            loss.backward()
        else:
            raise NotImplementedError(f"unsupported dtype for training: {dtype}")
        X, Y = get_batch(split = 'train', 
                 split_dataset = split_dataset, 
                 changer = changer, 
                 batch_size = batch_size,
                 update_interval = update_interval,
                 block_size = block_size,
                 device=device,
                 )
        

    if dtype == 'float16':  
        scaler.unscale_(optimizer)
    elif dtype == 'bfloat16' or dtype == 'float32':
        pass
    else:
        raise NotImplementedError(f"unsupported dtype for training: {dtype}")

    # 监控 clip 前的 gradnorm
    total_grad_norm_before = torch.norm(torch.stack([
        p.grad.detach().norm(2)
        for p in raw_model.parameters()
        if p.grad is not None
    ]), 2)
    emb_param = raw_model.transformer.wte.weight
    emb_grad_norm_before = (
        emb_param.grad.detach().norm(2)
        if emb_param.grad is not None else torch.tensor(float('nan'), device=emb_param.device)
    )

    # clip!
    returned_total_before = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

    # 监控 clip 后的 gradnorm
    total_grad_norm_after = torch.norm(torch.stack([
        p.grad.detach().norm(2)
        for p in raw_model.parameters()
        if p.grad is not None
    ]), 2)
    emb_grad_norm_after = (
        emb_param.grad.detach().norm(2)
        if emb_param.grad is not None else torch.tensor(float('nan'), device=emb_param.device)
    )
    if master_process:
        writer.add_scalar('GradNorm/total_before_clip', float(total_grad_norm_before), iter_num)
        writer.add_scalar('GradNorm/total_after_clip',  float(total_grad_norm_after),  iter_num)
        writer.add_scalar('GradNorm/embedding_before_clip', float(emb_grad_norm_before), iter_num)
        writer.add_scalar('GradNorm/embedding_after_clip',  float(emb_grad_norm_after),  iter_num)
    # --- 梯度监控到此结束 ---

    if dtype == 'float16':
        scaler.step(optimizer)
        scaler.update()
    elif dtype == 'bfloat16' or dtype == 'float32':
        optimizer.step()
    else:
        raise NotImplementedError(f"unsupported dtype for training: {dtype}")
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if (iter_num + 1) % log_interval == 0 and master_process:
        lossf = accum_loss_cpu
        prev_micro_loss = lossf / gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%") # , idx of selected txt {idx}"
        
        # --- TensorBoard: Log loss and MFU ---
        writer.add_scalar('Training Loss', float(lossf), iter_num)
        if local_iter_num >= 5:
            writer.add_scalar('MFU', running_mfu, iter_num)
        # --- End TensorBoard ---
    
    if record_interval > 0 and (iter_num + 1) % record_interval == 0:
        changer.get_frequency()
    
    if update_interval > 0 and (iter_num + 1) % update_interval == 0 and iter_num >= warmup_iters  and iter_num <= fix_iters:
        barrier()
        for _ in range(args.num_update):
            item_to_replace = changer.try_replace()
            if item_to_replace is not None:
                changer.updated_tokenizer()
                idx, x, y = item_to_replace
                embeddings = model.module.transformer.wte.weight
                new_embedding = (embeddings[x] + embeddings[y]) / 2
                with torch.no_grad():
                    embeddings[idx] = new_embedding
        if tok_to_compare is not None:
            jd = calc_jaccard_distance(changer.tokenizer, tok_to_compare)
            if master_process:
                writer.add_scalar('jaccard_distance', float(jd), iter_num)



    iter_num += 1
    local_iter_num += 1

    if iter_num % eval_interval == 0:
        barrier()
        if master_process:
            evaluate(model=model, 
                     split_dataset=split_dataset, 
                     changer=changer, 
                     ctx=ctx, 
                     batch_size=batch_size, 
                     eval_iters=eval_iters, 
                     iter_num=iter_num, 
                     update_interval=update_interval,
                     device=device,
                     writer=writer,
                    )

    if iter_num % save_interval == 0 or iter_num > max_iters:
        barrier()
        save_checkpoint(master_process, iter_num, out_dir, raw_model, optimizer, model_args, changer)
        if iter_num > max_iters:
            break

# iter_num = 0
# barrier()

# --- TensorBoard: Close the writer ---
if master_process:
    writer.close()
# --- End TensorBoard ---

if ddp:
    destroy_process_group()
