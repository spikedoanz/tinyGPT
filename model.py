from typing import Optional
from dataclasses import dataclass
import math

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
    seqlen: int = 1024
    vocab_size: int = 50304 # 768 * 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class FFN:
  def __init__(self, config: GPTConfig):
    self.wi = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
    self.act = lambda x: x.gelu()
    self.wo = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = lambda x: x.dropout(config.dropout) 

  def __call__(self, x: Tensor) -> Tensor:
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

  def __call__(self, x: Tensor, mask: Optional[Tensor]=None) -> Tensor:
    H, HS = self.n_head, self.n_embd // self.n_head
    shard = lambda x: x.rearrange('b s (h hs) -> b h s hs', h=H, hs=HS)
    drahs = lambda x: x.rearrange('b h s hs -> b s (h hs)', h=H, hs=HS)
    xq, xk, xv = [shard(x_) for x_ in (self.wq(x), self.wk(x), self.wv(x))]
    att = xq.scaled_dot_product_attention(xk, xv, mask)
    qk = Tensor.einsum("b h s d, b h t d -> b h s t", xq, xk) 
    qk = qk / math.sqrt(xq.shape[-1])
    if mask is not None: qk = qk + mask
    att = qk.softmax(-1).dropout(self.dropout) @ xv
    att = att.dropout(self.dropout)
    out = self.wo(drahs(att)).dropout(self.dropout)
    return out
    

class Block:
  def __init__(self, config: GPTConfig):
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.att = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = FFN(config)

  def __call__(self, x: Tensor, mask:Optional[Tensor]=None) -> Tensor:
    x = x + self.att(self.ln_1(x), mask)
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT:
  def __init__(self, config: GPTConfig):
    assert config.vocab_size is not None
    assert config.seqlen is not None
    self.config = config
    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
    self.wpe = nn.Embedding(config.seqlen, config.n_embd)
    self.drop = lambda x: x.dropout(config.dropout)
    self.blocks = [Block(config) for _ in range(config.n_layer)]
    self.ln = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

  def __call__(self, ts: Tensor) -> Tensor:
    t = ts.shape[-1]
    assert t <= self.config.seqlen, "token dim cannot be more than block size"
    pos = Tensor.arange(0, t, dtype=dtypes.long)
    tok_emb = self.wte(ts)  # b t   -> b t n_embd
    pos_emb = self.wpe(pos) # t     -> t n_embd
    x = self.drop(tok_emb + pos_emb)
    mask = (1 - Tensor.ones((t,t)).tril()) * -1e9

    for i, block in enumerate(self.blocks):
      x = block(x, mask)
    return self.lm_head(x)

if __name__ == "__main__":
  model = GPT(GPTConfig())
  B = 1
  S = 1024
  x = Tensor.randint((B,S))
  print(x.shape)
  out = model(x)
  print(out.shape) # B S T
  target = Tensor.randint((B,S), high=GPTConfig().vocab_size)
  loss = out.reshape(-1, GPTConfig().vocab_size).cross_entropy(target.reshape(-1))
  print(loss.shape, loss.item()) # should be ln(T)
