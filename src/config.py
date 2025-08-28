from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_id: str
    dtype: str = 'fp16'
    attn_impl: str = 'sdpa'
    quant: str = 'none'
    low_cpu_mem_usage: bool = True

@dataclass
class BenchConfig:
    batch: int = 1
    warmup: int = 3
    iters: int = 10

@dataclass
class TestSpec:
    kind: str
    length: int
    ubatch: int = 0
