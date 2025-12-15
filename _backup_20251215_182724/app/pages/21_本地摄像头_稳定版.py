import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")  # 各页面第一行

from app.utils.i18n import t

st.title(t("webcam_title"))
st.caption(t("webcam_tip"))

st.info("📌 占位页面：此处接入你的本地摄像头深蹲识别逻辑。\n\n"
        "如果你使用 OpenCV：请先安装 `pip install opencv-python`，并在此处编写读取/推理/展示的循环。")
# TODO: 在这里粘贴你原本“第三个能识别深蹲”的摄像头推理代码
