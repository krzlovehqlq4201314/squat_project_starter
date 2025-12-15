import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from app.vis_utils import build_heatmap, colorize_heatmap, overlay_rgba_on_bgr, draw_skeleton_rgba

st.set_page_config(page_title="webcam cv2", layout="wide")

# 侧边栏参数
th = st.sidebar.slider("判定为 squat 的阈值", 0.0, 1.0, 0.5, 0.01)
show_heatmap = st.sidebar.checkbox("显示热力图", value=False)
show_skeleton = st.sidebar.checkbox("显示关键线", value=True)
heat_alpha = st.sidebar.slider("热力图透明度", 0.0, 1.0, 0.45, 0.01)
width = st.sidebar.selectbox("分辨率宽", [480, 640, 800, 960, 1280], index=1)
max_fps = st.sidebar.slider("最大FPS", 5, 30, 15, 1)

start = st.button("▶ 开始")
stop_placeholder = st.empty()
frame_ph = st.empty()

if start:
    stop_btn = stop_placeholder.button("■ 停止")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 // 16))
    cap.set(cv2.CAP_PROP_FPS, max_fps)

    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    last = 0.0
    try:
        while True:
            if stop_btn:
                break
            ok, frame = cap.read()
            if not ok:
                st.warning("未读取到摄像头帧")
                break
            # OpenCV 是BGR，MediaPipe 需要RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_pose.process(rgb)

            H, W = frame.shape[:2]
            pts = []
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    pts.append((lm.x * W, lm.y * H))
            else:
                pts = []

            # 叠加可视化（防崩版）
            try:
                if show_heatmap and pts:
                    heat = build_heatmap(frame.shape, pts, sigma=14, intensity=1.1)
                    frame = overlay_rgba_on_bgr(frame, colorize_heatmap(heat), alpha=heat_alpha)
                if show_skeleton and pts:
                    frame = draw_skeleton_rgba(frame, pts, thickness=2)
            except cv2.error as e:
                # OpenCV 异常兜底，避免整页崩溃
                st.warning(f"可视化异常：{e}")

            # 这里你自己的分类/阈值判定逻辑……
            # demo：简单根据膝关节夹角/或某个概率占位
            # prob = some_model(...); label = "squat" if prob >= th else "not_squat"

            frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # 简单限帧
            now = time.time()
            dt = now - last
            min_dt = 1.0 / max_fps
            if dt < min_dt:
                time.sleep(min_dt - dt)
            last = now
    finally:
        cap.release()
        stop_placeholder.empty()
