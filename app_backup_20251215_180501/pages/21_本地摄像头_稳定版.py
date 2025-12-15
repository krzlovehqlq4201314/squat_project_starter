import streamlit as st
from app.i18n import t, language_selector

language_selector("sidebar", key="lang_webcam")

st.title(t("webcam_title"))
# 在此下方继续你的本地摄像头业务代码……
