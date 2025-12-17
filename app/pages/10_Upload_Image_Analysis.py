import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from utils.pose import analyze

st.set_page_config(page_title="Upload Image • Squat Checker", page_icon="🏋️", layout="wide")

st.title("Upload Image • Squat Checker")

if "pose_model" not in st.session_state:
    st.session_state.pose_model = YOLO("yolov8n-pose.pt")  # or your local path

draw_lines = st.checkbox("Draw skeleton lines", value=True)
draw_heat  = st.checkbox("Draw heatmap", value=True)

file = st.file_uploader("Drag & drop or click to upload (JPG/PNG)", type=["jpg","jpeg","png"])
if file:
    data = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    vis, msg = analyze(st.session_state.pose_model, img_bgr,
                       draw_lines=draw_lines, draw_heat=draw_heat, knee_ok_max=95)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True, caption=file.name)
    st.info(msg)
