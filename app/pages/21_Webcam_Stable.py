import streamlit as st
import cv2
from ultralytics import YOLO
from utils.pose import analyze

st.set_page_config(page_title="Webcam • Squat Checker", page_icon="📷", layout="wide")
st.title("Webcam • Squat Checker")

if "pose_model" not in st.session_state:
    st.session_state.pose_model = YOLO("yolov8n-pose.pt")

draw_lines = st.checkbox("Draw skeleton lines", value=True)
draw_heat  = st.checkbox("Draw heatmap", value=False)

cam = st.camera_input("Take a snapshot")
if cam is not None:
    file_bytes = cam.getvalue()
    img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    vis, msg = analyze(st.session_state.pose_model, img_array,
                       draw_lines=draw_lines, draw_heat=draw_heat, knee_ok_max=95)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True, caption="snapshot.jpg")
    st.info(msg)
