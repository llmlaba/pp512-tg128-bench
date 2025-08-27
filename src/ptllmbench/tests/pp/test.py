import torch
from src.ptllmbench.timing.timer import time_repeated, summarize_times
from src.ptllmbench.tests.base import TestResult

def _pp_once(model, x, attn, ubatch: int|None):
    seqlen = x.shape[1]
    if not ubatch or ubatch >= seqlen:
        _ = model(input_ids=x, attention_mask=attn, use_cache=True); return
    past = None; pos = 0; B = x.shape[0]
    while pos < seqlen:
        chunk = x[:, pos:pos+ubatch]
        total_ctx = pos + chunk.shape[1]
        attn_step = torch.ones(B, total_ctx, dtype=torch.long, device=x.device)
        out = model(input_ids=chunk, attention_mask=attn_step, use_cache=True, past_key_values=past)
        past = out.past_key_values
        pos += chunk.shape[1]

def run_pp(model, x, attn, warmup: int, iters: int, ubatch: int|None, device: str) -> TestResult:
    fn = lambda: _pp_once(model, x, attn, ubatch)
    times = time_repeated(fn, warmup, iters, device)
    t_med, t_mean, t_std = summarize_times(times)
    toks = x.numel()
    return TestResult(tps=toks/t_med, t_med=t_med, t_mean=t_mean, t_std=t_std)
