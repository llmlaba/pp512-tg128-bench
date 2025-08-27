import argparse
from rich.console import Console
import torch, transformers as _tf
from src.ptllmbench.version import VERSION
from src.ptllmbench.config import ModelConfig, BenchConfig, TestSpec
from src.ptllmbench.models.loader import ModelLoader
from src.ptllmbench.utils.gpu import detect_backend, device_string
from src.ptllmbench.utils.render import render_header, render_table, render_footer
from src.ptllmbench.tests.pp.data import make_random_prompt
from src.ptllmbench.tests.pp.test import run_pp
from src.ptllmbench.tests.tg.data import make_start_prompt
from src.ptllmbench.tests.tg.test import run_tg

def human_params(model) -> str:
    try:
        params = sum(p.numel() for p in model.parameters())
    except Exception:
        params = getattr(getattr(model, 'config', None), 'num_parameters', 0) or 0
    if params >= 1e9: return f"{params/1e9:.2f} B"
    if params >= 1e6: return f"{params/1e6:.2f} M"
    return str(params)

def parse_tests(test_args):
    specs = []
    for t in test_args:
        t = t.lower().strip()
        if t.startswith('pp'):
            n = int(t[2:]); specs.append(TestSpec(kind='pp', length=n))
        elif t.startswith('tg'):
            n = int(t[2:]); specs.append(TestSpec(kind='tg', length=n))
        else:
            raise ValueError(f"Unknown test '{t}' (expected ppN or tgN)")
    return specs

def main(argv=None):
    console = Console()
    ap = argparse.ArgumentParser(prog='pt-llm-bench', description='PyTorch LLM bench (llama-bench-like)')
    ap.add_argument('-m','--model', required=True)
    ap.add_argument('--tests', nargs='+', default=['pp512','tg128'])
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16','bf16','fp32'])
    ap.add_argument('--quant', type=str, default='none', choices=['none','4bit'])
    ap.add_argument('--attn', type=str, default='sdpa')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--ubatch', type=int, default=0)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        console.print('[yellow]WARNING[/]: CUDA/ROCm device not available; running on CPU will be slow.')

    model_cfg = ModelConfig(model_id=args.model, dtype=args.dtype, attn_impl=args.attn, quant=args.quant)
    bench_cfg = BenchConfig(batch=args.batch, warmup=args.warmup, iters=args.iters)
    tests = parse_tests(args.tests)

    backend = detect_backend(); device = device_string()
    render_header(console, torch.__version__, _tf.__version__, backend, device)

    loader = ModelLoader(model_cfg)
    tok, model, dev = loader.load()

    rows = []
    for spec in tests:
        if spec.kind == 'pp':
            x, attn = make_random_prompt(tok, bench_cfg.batch, spec.length, dev)
            res = run_pp(model, x, attn, warmup=bench_cfg.warmup, iters=bench_cfg.iters, ubatch=(args.ubatch or None), device=str(dev))
            tps = f"{res.tps:.2f} ± {res.t_std:.2f}"
            rows.append({'model': args.model, 'params': human_params(model), 'dtype': args.dtype, 'quant': args.quant, 'backend': backend, 'device': device[:22], 'batch': str(bench_cfg.batch), 'test': f'pp{spec.length}', 'tps': tps})
        elif spec.kind == 'tg':
            x0, attn0 = make_start_prompt(tok, bench_cfg.batch, dev)
            res = run_tg(model, x0, attn0, gen_len=spec.length, warmup=max(1, bench_cfg.warmup//2), iters=max(5, bench_cfg.iters//2), device=str(dev))
            tps = f"{res.tps:.2f} ± {res.t_std:.2f}"
            rows.append({'model': args.model, 'params': human_params(model), 'dtype': args.dtype, 'quant': args.quant, 'backend': backend, 'device': device[:22], 'batch': str(bench_cfg.batch), 'test': f'tg{spec.length}', 'tps': tps})
        else:
            raise RuntimeError('unknown test kind')

    render_table(console, rows)
    render_footer(console, torch.__version__, _tf.__version__)
    return 0
