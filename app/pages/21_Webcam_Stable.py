# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from inspect import signature
from app.utils.pose import analyze  # 仅复用你的分析逻辑，判定不改

st.set_page_config(page_title="Webcam – Squat", page_icon="🏋️", layout="wide")
st.title("Real-time Webcam Squat Check")

mirror = st.toggle("Mirror preview (front camera)", value=True)
photo = st.camera_input("Take a photo (include hips, knees and ankles in frame)")

# COCO-17 索引
NOSE=0; L_SH=5; R_SH=6; L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16
PAIRS_FULL  = [(L_SH, R_SH),(L_SH,L_HIP),(R_SH,R_HIP),(L_HIP,R_HIP),
               (L_HIP,L_KNEE),(L_KNEE,L_ANK),(R_HIP,R_KNEE),(R_KNEE,R_ANK)]

def _get_yolo(model_ref):
    key = f"_webcam_yolo_{str(model_ref)}"
    if key not in st.session_state:
        st.session_state[key] = YOLO(model_ref if isinstance(model_ref, str) else "weights/yolo11n-pose.pt")
    return st.session_state[key]

def _infer_keypoints(img_bgr, model_ref):
    model = _get_yolo(model_ref)
    r = model.predict(source=img_bgr, imgsz=640, verbose=False)[0]
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy)==0:
        return None, None
    xy  = r.keypoints.xy.cpu().numpy()   # (N,17,2)
    conf= r.keypoints.conf.cpu().numpy() # (N,17)
    # 取最大框的人
    idx = 0
    if r.boxes is not None and len(r.boxes)>1:
        xyxy = r.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])
        idx = int(np.argmax(areas))
    return xy[idx], conf[idx]

def _draw_heatmap(img, pts, conf, radius=18, alpha=0.45, th=0.2):
    over = img.copy()
    if pts is None: return img
    for i,(x,y) in enumerate(pts):
        if conf is None or conf[i] is None: 
            continue
        if conf[i] >= th:
            cv2.circle(over, (int(x),int(y)), radius, (0,255,255), -1)
    return cv2.addWeighted(over, alpha, img, 1-alpha, 0)

def _draw_pairs(img, pts, conf, pairs, th=0.25):
    out = img.copy()
    if pts is None or conf is None: 
        return img
    for a,b in pairs:
        if conf[a] is not None and conf[b] is not None and conf[a]>=th and conf[b]>=th:
            pa = (int(pts[a,0]), int(pts[a,1]))
            pb = (int(pts[b,0]), int(pts[b,1]))
            cv2.line(out, pa, pb, (0,255,0), 3)
    return out

def _lower_body_present(conf, th=0.35):
    if conf is None: return False
    need = [L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANK,R_ANK]
    return all(conf[i] is not None and conf[i] >= th for i in need)

def _call_analyze_auto(model_ref, img_bgr):
    # 兼容不同签名的 analyze，不改你的函数
    try:
        first = list(signature(analyze).parameters.keys())[0].lower()
        if first in {"model","model_path","pose_model","yolo","weights"}:
            return analyze(model_ref, img_bgr)
        return analyze(img_bgr, model_ref)
    except TypeError:
        try:
            return analyze(model_ref, img_bgr)
        except Exception:
            return analyze(img_bgr, model_ref)

if photo is not None:
    file_bytes = photo.getvalue()
    img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        from io import BytesIO
        from PIL import Image
        img_rgb = np.array(Image.open(BytesIO(file_bytes)).convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if mirror:
        img_bgr = cv2.flip(img_bgr, 1)

    if "pose_model" not in st.session_state:
        st.session_state.pose_model = "weights/yolo11n-pose.pt"

    # —— 先取关键点 —— #
    pts, conf = _infer_keypoints(img_bgr, st.session_state.pose_model)

    if pts is None:
        # 无人 -> 也画空热力图（不连线），提示
        vis = _draw_heatmap(img_bgr, None, None)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="snapshot.jpg", use_column_width=True)
        st.info("NOT SQUAT ❌ | No person detected.")
    elif not _lower_body_present(conf):
        # 下肢不完整 -> 画热力图 + “可用的”关键线（成对点都够阈值才连）
        vis = _draw_heatmap(img_bgr, pts, conf, radius=18, alpha=0.45, th=0.2)
        vis = _draw_pairs(vis, pts, conf, PAIRS_FULL, th=0.25)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="snapshot.jpg", use_column_width=True)
        st.info("NOT SQUAT ❌ | Lower body not visible. Please include hips, knees and ankles in the frame.")
    else:
        # 下肢齐全 -> 先按原逻辑 analyze 得到判定，再叠加热力图与关键线
        vis_bgr, msg = _call_analyze_auto(st.session_state.pose_model, img_bgr)
        vis_bgr = _draw_heatmap(vis_bgr, pts, conf, radius=14, alpha=0.35, th=0.2)
        vis_bgr = _draw_pairs(vis_bgr, pts, conf, PAIRS_FULL, th=0.25)
        st.image(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), caption="snapshot.jpg", use_column_width=True)
        st.info(msg)
else:
    st.caption("Tip: keep hips, knees and ankles fully visible in the frame; otherwise it will be NOT SQUAT.")
