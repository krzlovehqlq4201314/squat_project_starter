import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")  # 必须第一行

from app.utils.i18n import t, language_selector
language_selector(location="sidebar", key="lang_radio_upload")

st.title(t("upload_title"))
st.caption(t("upload_tip"))

uploaded = st.file_uploader(t("upload_prompt"), type=["jpg","jpeg","png"])
if uploaded is not None:
    try:
        from PIL import Image
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)
        st.success(t("upload_ok"))
        # TODO: 这里接你原来的“热力图/关键线”算法函数；没有的话先显示图片不报错
    except Exception as e:
        st.error(f"加载图片失败：{e}")
