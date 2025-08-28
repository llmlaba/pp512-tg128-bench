import torch

def detect_backend():
    hip = getattr(torch.version, 'hip', None)
    if hip:
        return f'ROCm {hip}'
    cu = getattr(torch.version, 'cuda', None)
    if cu:
        return f'CUDA {cu}'
    return 'CPU'

def detect_device_name(index: int = 0) -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(index)
        except Exception:
            return 'CUDA Device'
    return 'CPU'

def device_string() -> str:
    return detect_device_name(0)
