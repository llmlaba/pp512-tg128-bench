import torch
from src.timing.timer import time_repeated, summarize_times
from src.tests.base import TestResult

def run_tg(model, x0, attn0, gen_len: int, warmup: int, iters: int, device: str) -> TestResult:
    def timed_decode():
        out = model(input_ids=x0, attention_mask=attn0, use_cache=True)
        past = out.past_key_values
        last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        past_len = x0.size(1)
        for _ in range(gen_len):
            attn = torch.ones(x0.shape[0], past_len+1, dtype=torch.long, device=x0.device)
            out = model(input_ids=last, attention_mask=attn, use_cache=True, past_key_values=past)
            past = out.past_key_values
            last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            past_len += 1
    times = time_repeated(timed_decode, warmup, iters, device)
    t_med, t_mean, t_std = summarize_times(times)
    toks = x0.shape[0] * gen_len
    return TestResult(tps=toks/t_med, t_med=t_med, t_mean=t_mean, t_std=t_std)
