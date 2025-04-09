import os
import numpy as np
import pickle
from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.helpers import trange

from model import GPT, GPTConfig

#------------------------------------------------------------------------------
# meta
out_dir = "out"
eval_interval = 100
eval_iters = 10
init_from = ["scratch", "resume"][0]
# data
dataset = 'shakespeare_char'
batch_size = 64
ctx_len = 256
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
# optimizer
lr = 1e-4
steps = 1000
#------------------------------------------------------------------------------
# override configs from commandline or file
config_keys = [k for k,v in globals().items() 
               if not k.startswith("_") 
               and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
#------------------------------------------------------------------------------
# dataloader
data_dir = os.path.join('data', dataset)
def get_batch(split):
  binary = "train.bin" if split == "train" else "val.bin"
  data = np.memmap(os.path.join(data_dir, binary), dtype=np.uint16, mode='r')
  ix = Tensor.randint(batch_size, high=len(data)-ctx_len).tolist()
  x = Tensor([data[i:i+ctx_len] for i in ix])
  y = Tensor([data[i+1:i+1+ctx_len] for i in ix], dtype=dtypes.int64)
  return x, y

# derive vocab size from dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
  meta_vocab_size = meta['vocab_size']
  print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# start with model_args from command line
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, ctx_len=ctx_len,
                       bias=bias, vocab_size=None, dropout=dropout) 
if init_from == "scratch":
  print("initializing from scratch")
  if meta_vocab_size is None:
    print("defaulting to vocab size 50304")
  model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
  model = GPT(GPTConfig(**model_args))
else:
  print(f"resuming from checkpoint in {out_dir}")
  raise NotImplementedError


opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr)

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

eval_loss = float('nan')
for i in (t:=trange(steps)):
  loss = train_step().item()
  if i%eval_interval == 0: eval_loss = eval_step().item()
  t.set_description(f"loss: {loss:4.4f}, eval_loss: {eval_loss:4.4f}")
