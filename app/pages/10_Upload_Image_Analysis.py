import cv2, numpy as np, streamlit as st
from utils.pose import analyze

st.set_page_config(page_title="Upload Image · Analysis", layout="wide")
st.title("Upload Image · Analysis")
st.caption("Upload a JPG/PNG. Optionally draw skeleton and keypoint heatmap; we also judge correctness and give tips.")

draw_lines = st.checkbox("Draw skeleton lines", value=True)
draw_heat  = st.checkbox("Draw heatmap", value=True)
file = st.file_uploader("Drag & drop or click to upload (JPG/PNG)", type=["jpg","jpeg","png"])

if file:
    data = np.frombuffer(file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    vis, msg = analyze(img, draw_lines=draw_lines, draw_heat=draw_heat)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=file.name)
    if msg.startswith("[OK]"):
        st.success(msg)
    else:
        st.error(msg)
else:
    st.info("No image selected yet.")
