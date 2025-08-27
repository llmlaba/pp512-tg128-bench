import time, numpy as np, torch

def synchronize_if_needed(device: str) -> None:
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

def time_repeated(fn, warmup: int, iters: int, device: str):
    for _ in range(max(0, warmup)):
        fn(); synchronize_if_needed(device)
    times = []
    for _ in range(iters):
        synchronize_if_needed(device)
        t0 = time.perf_counter(); fn(); synchronize_if_needed(device)
        times.append(time.perf_counter() - t0)
    return times

def summarize_times(times):
    arr = np.array(times, dtype=float)
    return float(np.median(arr)), float(arr.mean()), float(arr.std())
