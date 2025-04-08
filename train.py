import os
import numpy as np
from tinygrad import Tensor, dtypes

#------------------------------------------------------------------------------
# IO
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits after first eval
always_save_checkpoint = True
init_from = ["scratch", "resume"][0]
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
# optimizer
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100
# system
dtype = dtypes.bfloat16
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
  ix = Tensor.randint(shape=batch_size,high=len(data)-block_size)

get_batch("train")

    
  
