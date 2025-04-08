from typing import Optional
from dataclasses import dataclass

from tinygrad import Tensor, nn, dtypes

"""
Shape index
B : batch
S : sequence length
E : embedding dimension
T : token dimension
"""

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class MLP:
  def __init__(self, config: GPTConfig):
    self.wi     = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
    self.act    = lambda x: x.gelu()
    self.wo     = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = lambda x: x.dropout(config.dropout) 

  def __call__(self, x: Tensor):
    x = self.wi(x)
    x = self.act(x)
    x = self.wo(x)
    x = self.dropout(x)
    return x


class CausalSelfAttention:
  def __init__(self, config: GPTConfig):
    assert config.n_embd % config.n_head == 0
    # TODO: kv caching
    self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout

  def __call__(self, x: Tensor, mask: Optional[Tensor]=None):
    # TODO: rope (?)
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    attn = xq.scaled_dot_product_attention(xk, xv, mask)
    attn = attn.dropout(self.dropout)
    out = self.wo(attn).dropout(self.dropout)
    print(out.shape)
    return out
    

class Block:
  def __init__(self, config: GPTConfig):
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def __call__(self, x: Tensor):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT:
  def __init__(self, config: GPTConfig):
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config
    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
    self.wpe = nn.Embedding(config.block_size, config.n_embd)
    self.drop = lambda x: x.dropout(config.dropout)
    self.blocks = [Block(config) for _ in range(config.n_layer)]
    self.ln = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

  def __call__(self, ts):
    t = ts.shape[-1]
    assert t <= self.config.block_size, "token dim cannot be more than block size"
    pos = Tensor.arange(0, t, dtype=dtypes.long)
    tok_emb = self.wte(ts)  # b t   -> b t n_embd
    pos_emb = self.wpe(pos) # t     -> t n_embd
    x = self.drop(tok_emb + pos_emb)
    for block in self.blocks:
      x = block(x)
    x = self.ln(x)
    return self.lm_head(x)

if __name__ == "__main__":
  model = GPT(GPTConfig())
  x = Tensor.randint((1,1024))
  model(x)
