import torch
import torch.fx as fx
import torch.nn as nn


import os
import math
import logging

from torch.nn import functional as F



logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPTSmallConfig(GPTConfig):
    """ GPT3-small like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class GPTMediumConfig(GPTConfig):
    """ GPT3-large like network roughly 350M params """
    n_layer = 24
    n_head = 16
    n_embd = 1024

class GPTLargeConfig(GPTConfig):
    """ GPT3-large like network roughly 760M params """
    n_layer = 24
    n_head = 16
    n_embd = 1536

class GPTXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 1.3B params """
    n_layer = 24
    n_head = 24
    n_embd = 2064

class GPTXXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 2.7B params """
    n_layer = 32
    n_head = 32
    n_embd = 2560

class GPTXXXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 6.7B params """
    n_layer = 32
    n_head = 32
    n_embd = 4096


class GPT13BConfig(GPTConfig):
    """ GPT3-XL like network roughly 13B params """
    n_layer = 48
    n_head = 48
    n_embd = 5184


class GPT175BConfig(GPTConfig):
    """ GPT3-XL like network roughly 175B params """
    n_layer = 96
    n_head = 96
    n_embd = 12288

class GPT1TConfig(GPTConfig):
    """ GPT3-XL like network roughly 1T params """
    n_layer = 128
    n_head = 128
    n_embd = 25600


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd={config.n_embd}, n_head={config.n_head}"
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.query = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.value = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # TODO: leave buffer on CPU for now, until we can do meta_tensor.to_empty()
        d = device if torch.device(device).type == "cuda" else "cpu"
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, device=d, dtype=dtype))
                 .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head

    def reset_parameters(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class EmbeddingStem(nn.Module):
    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        b, t = idx.size()
        #assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        return self.drop(token_embeddings + position_embeddings)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(
        self,
        config,
        device=None,
        dtype=torch.float32,
        wrapper=lambda m : m,
        version="pytorch",
        cpu_offload=False,
    ):
        super().__init__()
        if version == "pytorch" or not cpu_offload:
            self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.attn = wrapper(CausalSelfAttention(config, device=device, dtype=dtype))
            self.mlp = nn.Sequential(
                wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype)),
                nn.GELU(),
                wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype)),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            print("fairscale fsdp for block")
            self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu())
            self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu())
            self.attn = wrapper(CausalSelfAttention(config, device=device, dtype=dtype).cpu())
            self.mlp = nn.Sequential(
                wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype).cpu()),
                nn.GELU(),
                wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype).cpu()),
                nn.Dropout(config.resid_pdrop),
            )

    def reset_parameters(self):
        self.attn.reset_parameters()
        for _, m in self.named_modules():
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()

        # input embedding stem
        self.emb_stem = EmbeddingStem(config, device=device, dtype=dtype)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, device=device, dtype=dtype) for _ in range(config.n_layer)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

vocab_size = 500
block_size = 256
batch_size = 2
model = GPT(GPTSmallConfig(500, 256))
gm = fx.symbolic_trace(model)
#gm = fx.symbolic_trace(model, concrete_args={"idx": torch.randint(0, vocab_size, (batch_size, block_size))})

#gm.graph.print_tabular()

"""
Traceback (most recent call last):
  File "test_gpt.py", line 234, in <module>
    gm = fx.symbolic_trace(model)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 857, in symbolic_trace
    graph = tracer.trace(root, concrete_args)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 566, in trace
    self.create_node('output', 'output', (self.create_arg(fn(*args)),), {},
  File "test_gpt.py", line 229, in forward
    x = self.blocks(x)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 556, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 372, in call_module
    return forward(*args, **kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 552, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/raid/shenli/pytorch/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/raid/shenli/pytorch/torch/nn/modules/container.py", line 141, in forward
    input = module(input)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 556, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 372, in call_module
    return forward(*args, **kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 552, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/raid/shenli/pytorch/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "test_gpt.py", line 204, in forward
    x = x + self.attn(self.ln1(x))
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 556, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 372, in call_module
    return forward(*args, **kwargs)
  File "/raid/shenli/pytorch/torch/fx/_symbolic_trace.py", line 552, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/raid/shenli/pytorch/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "test_gpt.py", line 129, in forward
    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
TypeError: slice indices must be integers or None or have an __index__ method
"""