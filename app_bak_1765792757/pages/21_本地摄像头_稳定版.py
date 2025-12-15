import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")  # 必须第一行

from app.utils.i18n import t, language_selector
language_selector(location="sidebar", key="lang_radio_webcam")

st.title(t("webcam_title"))
st.info(t("webcam_tip"))

try:
    import cv2  # 若未安装，将走 except
    # TODO: 如需开启实时采集：cap = cv2.VideoCapture(0) ...（按你原逻辑补）
except Exception:
    st.warning("未检测到 OpenCV。请先执行：pip install opencv-python")
