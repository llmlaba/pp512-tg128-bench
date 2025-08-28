from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import ModelConfig

def parse_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ('fp16','float16'): return torch.float16
    if n in ('bf16','bfloat16'): return torch.bfloat16
    if n in ('fp32','float32'): return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')

class ModelLoader:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
    def load(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
        tok = AutoTokenizer.from_pretrained(self.cfg.model_id, use_fast=True)
        dtype = parse_dtype(self.cfg.dtype)
        attn_impl = self.cfg.attn_impl
        if self.cfg.quant == 'none':
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, torch_dtype=dtype, attn_implementation=attn_impl, low_cpu_mem_usage=self.cfg.low_cpu_mem_usage).to('cuda').eval()
            device = torch.device('cuda')
        elif self.cfg.quant == '4bit':
            from transformers import BitsAndBytesConfig
            qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.float16 if dtype==torch.float16 else torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, quantization_config=qconf, torch_dtype=dtype, attn_implementation=attn_impl, device_map='auto').eval()
            device = next(model.parameters()).device
        else:
            raise ValueError("quant must be 'none' or '4bit'")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        return tok, model, device
