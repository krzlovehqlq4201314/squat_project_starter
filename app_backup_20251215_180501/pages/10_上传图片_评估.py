import streamlit as st
from app.i18n import t, language_selector

language_selector("sidebar", key="lang_upload")

st.title(t("upload_title"))
# 在此下方继续你的上传/评估业务代码……
