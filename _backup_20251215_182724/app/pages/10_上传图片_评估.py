import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")  # 各页面第一行

from app.utils.i18n import t
from PIL import Image

st.title(t("upload_title"))
st.caption(t("upload_tip"))

uploaded = st.file_uploader(t("upload_prompt"), type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.success(t("upload_ok"))
    st.image(img, use_column_width=True)

# ---- 这里是你原先的角度/热力图/关键点逻辑的插槽 ----
# TODO: 把你之前的推理/可视化代码粘到这里（读取 img 变量即可）
