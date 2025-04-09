import os
import math
import pickle
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.helpers import trange, getenv

from model import GPT, GPTConfig

#-- config ---------------------------------------------------------------------
out_dir         = getenv("OUT_DIR", "out")
chkpt           = getenv("CHECKPOINT", False)
chkpt_fn        = getenv("CHECKPOINT_PATH", "model.safetensors")
chkpt_interval  = getenv("CHECKPOINT_INTERVAL", 10)
eval_interval   = getenv("EVAL_INTERVAL", chkpt_interval)
eval_iters      = getenv("EVAL_ITERS", 2)
# data
dataset         = getenv("DATASET", "shakespeare_char")
batch_size      = getenv("BS", 128)
seqlen          = getenv("SEQLEN", 256)
# model
n_layer         = getenv("N_LAYER", 6)
n_head          = getenv("N_HEAD" , 6)
n_embd          = getenv("N_EMBD" , 384)
dropout         = getenv("DROPOUT", 0.2)
bias            = getenv("BIAS", False)
# optimizer
max_lr          = getenv("MAX_LR", 1e-3)
min_lr          = getenv("MIN_LR", 1e-4)
steps           = getenv("STEPS", 1000)
warmup_steps    = getenv("WARMUP_STEPS", steps // 10)
lr_decay_steps  = getenv("LR_DECAY_STEPS", steps)
weight_decay    = getenv("WEIGHT_DECAY", 1e-1)
beta1           = getenv("BETA1", 0.9)
beta2           = getenv("BETA2", 0.99)

#-- dataloader -----------------------------------------------------------------
data_dir = os.path.join('data', dataset)
def get_batch(split):
  binary = "train.bin" if split == "train" else "val.bin"
  data = np.memmap(os.path.join(data_dir, binary), dtype=np.uint16, mode='r')
  ix = Tensor.randint(batch_size, high=len(data)-seqlen).tolist()
  x = Tensor([data[i:i+seqlen] for i in ix])
  y = Tensor([data[i+1:i+1+seqlen] for i in ix], dtype=dtypes.int64)
  return x, y

#-- model setup ----------------------------------------------------------------
# derive vocab size from dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
  meta_vocab_size = meta['vocab_size']
  print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# start with model_args from command line
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seqlen=seqlen,
                       bias=bias, vocab_size=None, dropout=dropout) 
if meta_vocab_size is None: print("defaulting to vocab size 50304")
model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
model = GPT(GPTConfig(**model_args))

#-- train utils ---------------------------------------------------------------
if getenv("RESUME"):
  chkpt_path = os.path.join(out_dir, chkpt_fn)
  nn.state.load_state_dict(model, nn.state.safe_load(chkpt_path))
  print(f"resuming from chkpt in {chkpt_path}")

def chkpt(fn="model.safetensors", step=0) -> str:
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    nn.state.safe_save(nn.state.get_state_dict(model), os.path.join(out_dir,fn))
    return f"step {step}:\t chkpt saved to {os.path.join(out_dir,fn)}"

def get_lr(it):
  if it < warmup_steps: return max_lr * (it+1) / (warmup_steps+1)
  if it > lr_decay_steps: return min_lr
  decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff in [0..1]
  return min_lr + coeff * (max_lr - min_lr)

opt = nn.optim.AdamW(
        nn.state.get_parameters(model), 
        lr=max_lr,
        b1=beta1,
        b2=beta2,
        weight_decay=weight_decay)

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  opt.zero_grad()
  x, y = get_batch("train")
  loss = model(x).rearrange("b s t -> (b s) t").cross_entropy(y.reshape(-1)).backward()
  opt.step()
  return loss

@TinyJit
@Tensor.test()
def eval_step() -> Tensor:
  losses = []
  for _ in range(eval_iters):
    x, y = get_batch("eval")
    losses.append(model(x).rearrange("b s t -> (b s) t").cross_entropy(y.reshape(-1)))
  return Tensor.stack(*losses).mean()

#-- print out config ----------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
for k in config: print(f"{k} = {config[k]}")

#-- train loop ----------------------------------------------------------------
eval_loss = float('nan')
for i in (t:=trange(steps)):
  opt.lr = get_lr(i)
  loss = train_step().item()
  if (i+1)%eval_interval == 0: eval_loss = eval_step().item()
  if chkpt and (i+1)%chkpt_interval == 0: t.write(chkpt(chkpt_fn,i))
  t.set_description(f"loss: {loss:4.4f}, eval_loss: {eval_loss:4.4f}")
