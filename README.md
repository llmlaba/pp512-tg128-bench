# pp512-tg128-bench
Python pp512 tg128 benchmark

## Reqirments
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 200TB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Python 3.12/3.11

## Test environment
- My test environment: HP Z440 + AMD Mi50

## Preparation

### Create virtualenv
- For AMD ROCm 6
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_bench
source ./.venv_llm_bench/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers>=4.41 accelerate einops
```

- For NVIDIA CUDA 12
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_bench
source ./.venv_llm_bench/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.41 accelerate einops
```

- Check pytorch
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```

### Get the Mistral
```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```

### Get benchmark source code
```bash
git clone https://github.com/llmlaba/pp512-tg128-bench.git
```

### Run test

```
python ./app.py -m ../mistral --tests pp512 tg128 --dtype fp16 --batch 2 --attn sdpa --warmup 3 --iters 10 --ubatch 128
```

## More test options

- [pp512](./pp512.md)
- [tg128](./tg128.md)
