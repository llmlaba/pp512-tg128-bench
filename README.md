# pp512-tg128-bench
Python pp512 tg128 benchmark


- prep
```
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_bench
source ./.venv_llm_bench/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

pip install transformers>=4.41 accelerate einops
```

- check pytorch
```
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.hip);print(torch.cuda.get_device_name(0));"
```

- run
```
python ./bench.py -m ./mistral --tests pp512 tg128 --dtype fp16 --batch 1 --attn sdpa
```