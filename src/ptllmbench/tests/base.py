from dataclasses import dataclass
import torch

@dataclass
class TestResult:
    tps: float
    t_med: float
    t_mean: float
    t_std: float

def make_attention_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)

def replace_special_ids(x: torch.Tensor, tok) -> None:
    special = set(tok.all_special_ids or [])
    if special:
        spec = torch.tensor(list(special), device=x.device)
        mask = torch.isin(x, spec)
        while bool(mask.any().item()):
            x[mask] = torch.randint(0, tok.vocab_size, (int(mask.sum()),), device=x.device)
            mask = torch.isin(x, spec)
    if tok.bos_token_id is not None:
        x[:, 0] = tok.bos_token_id
