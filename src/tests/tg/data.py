import torch
from src.tests.base import make_attention_mask

def make_start_prompt(tok, batch: int, device):
    bos = tok.bos_token_id or 1
    x0 = torch.full((batch, 1), bos, dtype=torch.long, device=device)
    attn0 = make_attention_mask(x0)
    return x0, attn0
