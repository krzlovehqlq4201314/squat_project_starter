@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
streamlit run app/main.py
