import cv2, numpy as np, streamlit as st
from utils.pose import analyze

st.set_page_config(page_title="Webcam Stable", layout="wide")
st.title("Webcam · Stable")

draw_lines = st.checkbox("Draw skeleton lines", value=True)
draw_heat  = st.checkbox("Draw heatmap", value=True)
photo = st.camera_input("Take Photo")

if photo:
    data = np.frombuffer(photo.getvalue(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    vis, msg = analyze(img, draw_lines=draw_lines, draw_heat=draw_heat)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="snapshot.jpg")
    if msg.startswith("[OK]"):
        st.success(msg)
    else:
        st.error(msg)
else:
    st.info("This page keeps a stable capture flow; click **Take Photo** to evaluate.")
