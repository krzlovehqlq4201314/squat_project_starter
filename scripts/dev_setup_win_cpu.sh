#!/usr/bin/env bash
set -e
python -m venv .venv311
source ./.venv311/Scripts/activate

python -m pip install --upgrade pip
# 先装 PyTorch CPU 版
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu torchvision==0.18.1+cpu

# 再装其余依赖
pip install -r requirements.txt

# 锁一次你本机上的版本（可选）
pip freeze > requirements-locked.txt

echo "OK. Activate: source ./.venv311/Scripts/activate"
echo "Run app:     streamlit run app/app.py"
