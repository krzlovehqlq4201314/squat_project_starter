import streamlit as st
st.set_page_config(page_title="Squat Checker (Demo)", layout="wide")
st.title("Squat Checker (Demo)")
st.markdown(
"""
**Welcome!**  
Use the sidebar to open:
- **Upload Image · Analysis** — run pose detection on a single JPG/PNG, with skeleton & heatmap overlays **and** textual feedback.
- **Webcam Stable** — take a snapshot via your webcam and analyze it.

This demo uses **Ultralytics YOLOv8-Pose (CPU)**. On first run, weights will be downloaded automatically.
"""
)
