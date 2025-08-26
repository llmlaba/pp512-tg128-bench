#!/usr/bin/env python3
import argparse, time, math, os, sys, json
from dataclasses import dataclass
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def human_params(n):
    if n >= 1e9: return f"{n/1e9:.2f} B"
    if n >= 1e6: return f"{n/1e6:.2f} M"
    return str(n)

def detect_backend():
    # PyTorch на ROCm все равно репортит 'cuda' девайс; различаем по версии
    hip = getattr(torch.version, "hip", None)
    if hip: 
        return f"ROCm {hip}"
    cu = getattr(torch.version, "cuda", None)
    if cu:
        return f"CUDA {cu}"
    return "CPU"

def detect_device_name():
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA Device"
    return "CPU"

def parse_dtype(s):
    s = s.lower()
    if s == "fp16" or s == "float16": return torch.float16
    if s == "bf16" or s == "bfloat16": return torch.bfloat16
    if s == "fp32" or s == "float32": return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")

def make_rand_batch(tok, batch, seqlen, device):
    vocab = tok.vocab_size
    x = torch.randint(0, vocab, (batch, seqlen), dtype=torch.long, device=device)
    # избегаем спец-идов
    special = set(tok.all_special_ids or [])
    if special:
        spec = torch.tensor(list(special), device=device)
        mask = torch.isin(x, spec)
        while bool(mask.any().item()):
            x[mask] = torch.randint(0, vocab, (int(mask.sum()),), device=device)
            mask = torch.isin(x, spec)
    if tok.bos_token_id is not None:
        x[:, 0] = tok.bos_token_id
    attn = torch.ones_like(x)
    return x, attn

@dataclass
class RunCfg:
    model_id: str
    batch: int
    pp_len: int
    tg_len: int
    dtype: torch.dtype
    quant: str
    attn_impl: str
    warmup: int
    iters: int
    ubatch: int|None

def load_model(cfg: RunCfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    quant = cfg.quant.lower()
    attn_impl = cfg.attn_impl
    if quant == "none":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        device = torch.device("cuda")
    elif quant == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            print("ERROR: bitsandbytes/transformers quantization not available. Install bitsandbytes.", file=sys.stderr)
            raise
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16 if cfg.dtype==torch.float16 else torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=qconf,
            torch_dtype=cfg.dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        ).eval()
        device = next(model.parameters()).device
    else:
        raise ValueError("quant must be one of: none, 4bit")
    return tok, model, device

@torch.inference_mode()
def bench_pp(model, x, attn, warmup=3, iters=10, ubatch=None, device="cuda"):
    # прогрев
    for _ in range(max(0, warmup)):
        _run_pp(model, x, attn, ubatch, device)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
    # замер
    times=[]
    for _ in range(iters):
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_pp(model, x, attn, ubatch, device)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
    t_med = float(np.median(times))
    toks = x.numel()
    return dict(tps=toks/t_med, t_med=t_med, t_mean=float(np.mean(times)), t_std=float(np.std(times)))

def _run_pp(model, x, attn, ubatch, device):
    seqlen = x.shape[1]
    if not ubatch or ubatch >= seqlen:
        _ = model(input_ids=x, attention_mask=attn, use_cache=True)
        return
    # микробатчинг (эмуляция n_ubatch): наращиваем KV-кэш
    past = None
    pos = 0
    B = x.shape[0]
    while pos < seqlen:
        chunk = x[:, pos:pos+ubatch]
        # внимание длиной total_ctx = pos + chunk_len
        total_ctx = pos + chunk.shape[1]
        attn_step = torch.ones(B, total_ctx, dtype=torch.long, device=chunk.device)
        out = model(input_ids=chunk, attention_mask=attn_step, use_cache=True, past_key_values=past)
        past = out.past_key_values
        pos += chunk.shape[1]

@torch.inference_mode()
def bench_tg(model, x0, attn0, gen_len, warmup=3, iters=5, device="cuda"):
    # прогрев (prefill + небольшой decode)
    for _ in range(max(0, warmup)):
        _ = model(input_ids=x0, attention_mask=attn0, use_cache=True)
        if device != "cpu" and torch.cuda.is_available(): torch.cuda.synchronize()
        past = None
        out = model(input_ids=x0, attention_mask=attn0, use_cache=True, past_key_values=past)
        past = out.past_key_values
        last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if device != "cpu" and torch.cuda.is_available(): torch.cuda.synchronize()
    # измерение только decode N токенов
    times=[]
    for _ in range(iters):
        if device != "cpu" and torch.cuda.is_available(): torch.cuda.synchronize()
        # префилл (не учитываем во времени tg)
        out = model(input_ids=x0, attention_mask=attn0, use_cache=True)
        past = out.past_key_values
        last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if device != "cpu" and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _step in range(gen_len):
            attn = torch.ones(x0.shape[0], x0.shape[1]+_step+1, dtype=torch.long, device=x0.device)
            out = model(input_ids=last, attention_mask=attn, use_cache=True, past_key_values=past)
            past = out.past_key_values
            last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if device != "cpu" and torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
    t_med = float(np.median(times))
    toks = x0.shape[0]*gen_len
    return dict(tps=toks/t_med, t_med=t_med, t_mean=float(np.mean(times)), t_std=float(np.std(times)))

def table_line(cols, widths):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"

def main():
    ap = argparse.ArgumentParser(description="PyTorch LLM bench (llama.cpp-like output)")
    ap.add_argument("-m","--model", required=True, help="HF model id or local path")
    ap.add_argument("--tests", nargs="+", default=["pp512","tg128"], help="one or more of: pp<N>, tg<N> (e.g., pp512 tg128)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--quant", type=str, default="none", choices=["none","4bit"])
    ap.add_argument("--attn", type=str, default="sdpa", help="sdpa or flash_attention_2 (if installed)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--ubatch", type=int, default=0, help="micro-batch tokens for pp (0 = disabled)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm device not available; running on CPU may be extremely slow.", file=sys.stderr)

    cfg = RunCfg(
        model_id=args.model,
        batch=args.batch,
        pp_len=512, tg_len=128,
        dtype=parse_dtype(args.dtype),
        quant=args.quant,
        attn_impl=args.attn,
        warmup=args.warmup,
        iters=args.iters,
        ubatch=args.ubatch if args.ubatch>0 else None,
    )

    bk = detect_backend()
    devname = detect_device_name()
    print(f"pytorch-llm-bench: torch={torch.__version__} transformers={{}} backend={bk}".format(__import__("transformers").__version__))
    print(f"device: {devname}")
    print()

    tok, model, device = load_model(cfg)

    # попытка подсчитать параметры
    try:
        params = sum(p.numel() for p in model.parameters())
    except Exception:
        params = getattr(getattr(model, "config", None), "num_parameters", 0) or 0

    # заголовок таблицы (похоже на llama-bench)
    headers = ["model", "params", "dtype", "quant", "backend", "device", "batch", "test", "t/s"]
    widths  = [28, 10, 6, 6, 10, 22, 5, 7, 16]
    print(table_line(headers, widths))
    print(table_line(["-"*w for w in widths], widths))

    for t in args.tests:
        t = t.lower()
        if t.startswith("pp"):
            try:
                n = int(t[2:])
            except:
                raise ValueError("pp test must be like pp512")
            x, attn = make_rand_batch(tok, cfg.batch, n, device)
            # прогон
            res = bench_pp(model, x, attn, warmup=cfg.warmup, iters=cfg.iters, ubatch=cfg.ubatch, device=str(device))
            tps = f"{res['tps']:.2f} ± {res['t_std']:.2f}"
            row = [cfg.model_id, human_params(params), args.dtype, args.quant, bk[:9], devname[:22], cfg.batch, f"pp{n}", tps]
            print(table_line(row, widths))

        elif t.startswith("tg"):
            try:
                n = int(t[2:])
            except:
                raise ValueError("tg test must be like tg128")
            # стартуем с короткого промпта (1 токен BOS)
            x0 = torch.full((cfg.batch, 1), tok.bos_token_id or 1, dtype=torch.long, device=device)
            attn0 = torch.ones_like(x0)
            res = bench_tg(model, x0, attn0, gen_len=n, warmup=max(1, cfg.warmup//2), iters=max(5, cfg.iters//2), device=str(device))
            tps = f"{res['tps']:.2f} ± {res['t_std']:.2f}"
            row = [cfg.model_id, human_params(params), args.dtype, args.quant, bk[:9], devname[:22], cfg.batch, f"tg{n}", tps]
            print(table_line(row, widths))

        else:
            raise ValueError("Unknown test: " + t)

    print()
    # как в llama-bench: краткая строка о версиях/сборке
    print(f"build: torch {torch.__version__}, transformers {__import__('transformers').__version__}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
