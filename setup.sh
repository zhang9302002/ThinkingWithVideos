# create env
# python 310 is a must, or there will be some dependency errors

conda create -n verl python=3.10
conda activate verl

# 1. install vllm and torch
pip install "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata
# 2. install transformers
pip install "transformers[hf_xet]==4.51.1" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy  pre-commit ruff 
# 3. install other dependencies
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
# 4. install flash-attn open-cv
pip install flash-attn==2.7.4.post1 opencv-python opencv-fixer 
python -c "from opencv_fixer import AutoFix; AutoFix()"
# 5. install decord rouge_score matplotlib
pip install decord rouge_score matplotlib

cd verl
pip install --no-cache-dir --no-deps -e .

echo "Successfully installed all packages"

