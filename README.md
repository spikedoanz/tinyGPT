# tinyGPT

> nanoGPT, but [tinier](https://github.com/tinygrad/tinygrad)

---

# quick setup


Installation
```
git clone https://github.com/spikedoanz/tinyGPT
cd tinyGPT
uv venv .venv
uv pip install tinygrad numpy requests
```

Dataset prep
```
python data/shakespeare_char/prepare.py
```

Run
```
python train.py
```

---

# TODO

- correctness, then split to a gpt branch
- wandb
- sharding for ddp
- microtricks
    - gradient clipping

---

# crackpot

- moe for mha
- moe for ffn
- NoPE
- mla
- muon
- norm analysis
