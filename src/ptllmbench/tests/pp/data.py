import torch
from ptllmbench.tests.base import replace_special_ids, make_attention_mask

def make_random_prompt(tok, batch: int, seqlen: int, device):
    x = torch.randint(0, tok.vocab_size, (batch, seqlen), dtype=torch.long, device=device)
    replace_special_ids(x, tok)
    attn = make_attention_mask(x)
    return x, attn
