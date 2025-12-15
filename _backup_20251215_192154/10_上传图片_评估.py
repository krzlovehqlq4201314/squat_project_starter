import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import streamlit as st
from app.utils.i18n import t
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

st.markdown(f"# {t('upload_title') if callable(t) else '上传图片·评估'}")
st.info(t('upload_tip') if callable(t) else "在此上传一张 JPG/PNG，页面将展示关键线与热力图（示例）。")

file = st.file_uploader(t('upload_prompt') if callable(t) else "请选择一张图片", type=["jpg","jpeg","png"])
col1, col2 = st.columns(2)
with col1:
    flag_skel = st.checkbox("叠加关键线", value=True)
with col2:
    flag_heat = st.checkbox("叠加热力图", value=True)

if file is not None:
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        res = pose.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    vis = img_np.copy()

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        pts = [(int(p.x*w), int(p.y*h), p.visibility) for p in lm]

        if flag_skel:
            for i, j in mp_pose.POSE_CONNECTIONS:
                xi, yi, vi = pts[i]
                xj, yj, vj = pts[j]
                if vi > 0.2 and vj > 0.2:
                    cv2.line(vis, (xi, yi), (xj, yj), (0, 255, 0), 2)
            for x, y, v in pts:
                if v > 0.2: cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)

        if flag_heat:
            heat = np.zeros((h, w), dtype=np.float32)
            r = max(3, int(min(h, w) * 0.02))
            for x, y, v in pts:
                if v > 0.2:
                    cv2.circle(heat, (x, y), r, 1.0, -1)
            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=min(h, w)*0.02)
            if heat.max() > 0:
                heat = (heat / heat.max() * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                vis = cv2.addWeighted(vis, 0.75, heat_color, 0.45, 0)

        st.success(t('upload_ok') if callable(t) else "图片上传成功")
        st.image(vis, caption=file.name)
    else:
        st.warning(t('no_person_detected') if callable(t) else "未检测到人体姿态，请换一张更清晰的全身图。")
