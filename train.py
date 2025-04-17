import os
import math
import pickle
import datetime

import wandb

import numpy as np 
from tinygrad import Tensor, nn, dtypes, TinyJit, Device
from tinygrad.nn import state
from tinygrad.helpers import trange, getenv

from model import GPT, GPTConfig


#-- config ---------------------------------------------------------------------
# wandb
use_wandb       = getenv("WANDB", 0)
project_name    = getenv("PROJECT_NAME", "tinygpt")
name            = getenv("NAME", datetime.datetime.now().strftime("%Y-%m-%d:%H-%M"))
# meta
out_dir         = getenv("OUT_DIR", "out")
chkpt           = getenv("CHECKPOINT", 0)
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
max_lr          = getenv("MAX_LR",  1e-3) # QUESTION: why do we do this?
min_lr          = getenv("MIN_LR",  1e-6)
start_lr        = getenv("LR",      1e-4) 
steps           = getenv("STEPS", 1000)
warmup_steps    = getenv("WARMUP_STEPS", steps // 10)
lr_decay_steps  = getenv("LR_DECAY_STEPS", steps)
weight_decay    = getenv("WEIGHT_DECAY", 1e-1)
beta1           = getenv("BETA1", 0.9)
beta2           = getenv("BETA2", 0.99)
grad_clip       = getenv("GRAD_CLIP", 10.0) # QUESTION: can't this be solved with math

_chkpt_path     = os.path.join(out_dir, chkpt_fn)


#-- ddp ------------------------------------------------------------------------
DDP             = getenv("DDP", 0)
GPUS            = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))

#-- dataloader -----------------------------------------------------------------
_data_dir = os.path.join('data', dataset)
def get_batch(split):
  binary = "train.bin" if split == "train" else "val.bin"
  data = np.memmap(os.path.join(_data_dir, binary), dtype=np.uint16, mode='r')
  ix = Tensor.randint(batch_size, high=len(data)-seqlen).tolist()
  x = Tensor([data[i:i+seqlen] for i in ix], dtype=dtypes.int64)
  y = Tensor([data[i+1:i+1+seqlen] for i in ix], dtype=dtypes.int64)
  # DDP: shard data along batch
  if DDP > 0: x, y = [t.shard_(GPUS, axis=0) for t in (x,y)] 
  return x, y

#-- model setup ----------------------------------------------------------------
# derive vocab size from dataset
_meta_path = os.path.join(_data_dir, 'meta.pkl')
_meta_vocab_size = None
if os.path.exists(_meta_path):
  with open(_meta_path, 'rb') as f:
    meta = pickle.load(f)
  _meta_vocab_size = meta['vocab_size']
  print(f"found vocab_size = {_meta_vocab_size} (inside {_meta_path})")
# start with model_args from command line
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seqlen=seqlen,
                       bias=bias, vocab_size=None, dropout=dropout) 
if _meta_vocab_size is None: print("defaulting to vocab size 50304")
model_args["vocab_size"] = _meta_vocab_size if _meta_vocab_size is not None else 50304
model = GPT(GPTConfig(**model_args))
# clone models to devices
if DDP > 0: {k: x.to_(GPUS) for k, x in nn.state.get_state_dict(model).items()}

#-- train utils ---------------------------------------------------------------

def lr_scheduler(it):
  if it < warmup_steps: return max_lr * (it+1) / (warmup_steps+1)
  if it > lr_decay_steps: return min_lr
  decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff in [0..1]
  return min_lr + coeff * (max_lr - min_lr)

opt = nn.optim.AdamW(
        state.get_parameters(model), 
        lr=start_lr,
        b1=beta1,
        b2=beta2,
        weight_decay=weight_decay)

def invert_dict(d): return {v: k for k, v in reversed(d.items())}
def dedup_dict(d): return invert_dict(invert_dict(d))
def get_train_state(model, optimizer):
  train_state = {"model": model, "optimizer": optimizer}
  return dedup_dict(state.get_state_dict(train_state))

def load_train_state(model, optimizer, state_dict):
  train_state = {"model": model, "optimizer": optimizer}
  big_dict = state.get_state_dict(train_state)
  dupe_names = {}
  for k, v in big_dict.items():
    if v not in dupe_names:
      dupe_names[v] = k
      assert k in state_dict
    state_dict[k] = state_dict[dupe_names[v]]

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  opt.zero_grad()
  x, y = get_batch("train")
  loss = model(x).rearrange("b s t -> (b s) t").cross_entropy(y.reshape(-1))
  loss.backward()
  #for p in opt.params: p.grad.clip(-grad_clip, grad_clip)
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
_config = {k: globals()[k] for k in config_keys} # will be useful for logging
for k in _config: print(f"{k} = {_config[k]}")

#-- train loop ----------------------------------------------------------------
if getenv("RESUME"):
  train_state = state.safe_load(_chkpt_path)
  train_state = load_train_state(model, opt, train_state)
  print(f"loaded training state from {_chkpt_path}")

if use_wandb: run = wandb.init(name=name, project=project_name,config=_config)
best_iter, best_eval_loss, eval_loss = 0, 1e9, float('nan')
for i in (t:=trange(steps)):
  lr = lr_scheduler(i)
  loss = train_step().item()
  if (i+1)%eval_interval == 0: eval_loss = eval_step().item()
  if (chkpt > 0) and (i+1)%chkpt_interval==0: 
    state.safe_save(get_train_state(model, opt), _chkpt_path)
    best_iter = i
  if use_wandb: wandb.log(
    {"train_loss": loss, "eval_loss": eval_loss,
     "learning_rate": lr_scheduler(i)}
    # TODO: gradient norm, mfu
    )
  t.set_description(f"loss: {loss:4.4f}, eval_loss: {eval_loss:4.4f}, last_checkpoint: {best_iter}")
if use_wandb: wandb.finish()
