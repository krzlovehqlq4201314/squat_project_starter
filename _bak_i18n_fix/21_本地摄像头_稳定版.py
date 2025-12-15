import os, sys
sys.path.append(os.path.dirname(__file__))
from i18n import t, select_language
from utils.i18n import t, select_language
import time, math
import cv2, numpy as np, streamlit as st
try:
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import mediapipe as mp
mp_pose = mp.solutions.pose

def angle_3pts(a,b,c):
    a=np.array(a,dtype=np.float32); b=np.array(b,dtype=np.float32); c=np.array(c,dtype=np.float32)
    ba=a-b; bc=c-b
    if np.linalg.norm(ba)<1e-6 or np.linalg.norm(bc)<1e-6: return None
    cosang=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    cosang=float(np.clip(cosang,-1.0,1.0))
    return float(np.degrees(np.arccos(cosang)))

def draw_skeleton_rgba(h,w,pts_xy,thick=2):
    rgba=np.zeros((h,w,4),dtype=np.uint8)
    if pts_xy is None: return rgba
    pairs=[(11,13),(13,15),(12,14),(14,16),(11,12),(23,24),(11,23),(12,24),(23,25),(25,27),(27,31),(24,26),(26,28),(28,32)]
    for a,b in pairs:
        if a<len(pts_xy) and b<len(pts_xy):
            pa,pb=pts_xy[a],pts_xy[b]
            if pa is not None and pb is not None:
                cv2.line(rgba,tuple(map(int,pa)),tuple(map(int,pb)),(0,255,255,255),thick,cv2.LINE_AA)
    for p in pts_xy:
        if p is not None:
            cv2.circle(rgba,tuple(map(int,p)),3,(255,255,0,255),-1,cv2.LINE_AA)
    return rgba

def make_heatmap_rgba(h,w,pts_xy,sigma=18,intensity=1.0):
    if pts_xy is None: return np.zeros((h,w,4),dtype=np.uint8)
    heat=np.zeros((h,w),dtype=np.float32)
    for p in pts_xy:
        if p is not None:
            x,y=map(int,p)
            if 0<=x<w and 0<=y<h: heat[y,x]+=1.0
    heat=cv2.GaussianBlur(heat,(0,0),sigmaX=sigma,sigmaY=sigma)
    if heat.max()>1e-6: heat=heat/heat.max()
    heat=(heat*255*np.clip(float(intensity),0.0,2.0)).clip(0,255).astype(np.uint8)
    colored=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
    rgba=cv2.cvtColor(colored,cv2.COLOR_BGR2BGRA)
    rgba[:,:,3]=heat
    return rgba

def overlay_rgba_on_bgr(bgr,rgba,alpha=0.45):
    if rgba is None or rgba.shape[:2]!=bgr.shape[:2]: return bgr
    out=bgr.astype(np.float32).copy()
    rgb=rgba[...,:3].astype(np.float32)
    a=(rgba[...,3:4].astype(np.float32)/255.0)*float(alpha)
    out=out*(1.0-a)+rgb*a
    return out.clip(0,255).astype(np.uint8)

# ---- sidebar ----
st.sidebar.header("参数 / Params")
cam_id = st.sidebar.selectbox(t("camera_id"), options=[0,1,2,3], index=0)
show_heat = st.sidebar.checkbox(t("show_heat"), value=False)
show_lines = st.sidebar.checkbox(t("show_lines"), value=True)
heat_alpha = st.sidebar.slider(t("alpha_label"), 0.0, 1.0, 0.45, 0.01)
width_opt = st.sidebar.selectbox(t("width_label"), [640,960,1280], index=0)
max_fps = st.sidebar.slider(t("fps_label"), 5, 30, 15, 1)
knee_thresh = st.sidebar.slider(t("knee_thresh"), 80, 140, 110, 1)

col_btn1, col_btn2 = st.columns([1,1])
start_clicked = col_btn1.button("▶ "+t("btn_start"), type="primary")
stop_clicked  = col_btn2.button("■ "+t("btn_stop"))

if 'running' not in st.session_state: st.session_state.running=False
if start_clicked: st.session_state.running=True
if stop_clicked:  st.session_state.running=False

st.title(t("camera_page_title"))
st.info(t("camera_info"))

frame_box = st.empty()
msg_box = st.empty()

if st.session_state.running:
    cap=cv2.VideoCapture(int(cam_id), cv2.CAP_DSHOW)
    if not cap.isOpened():
        msg_box.error(t("err_open_cam"))
        st.session_state.running=False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width_opt))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width_opt*9//16))
        cap.set(cv2.CAP_PROP_FPS, float(max_fps))

        pose=mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

        last=time.time()
        while st.session_state.running:
            ok,frame=cap.read()
            if not ok:
                msg_box.error(t("err_read_cam"))
                break

            h,w=frame.shape[:2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)

            pts_xy=None; left_knee=right_knee=None
            if res.pose_landmarks:
                lm=res.pose_landmarks.landmark
                pts_xy=[(int(p.x*w), int(p.y*h)) for p in lm]
                if len(pts_xy)>28:
                    left_knee  = angle_3pts(pts_xy[23],pts_xy[25],pts_xy[27])
                    right_knee = angle_3pts(pts_xy[24],pts_xy[26],pts_xy[28])

                if show_lines:
                    sk=draw_skeleton_rgba(h,w,pts_xy,2)
                    frame=overlay_rgba_on_bgr(frame,sk,alpha=max(heat_alpha,0.15))
                if show_heat:
                    hm=make_heatmap_rgba(h,w,pts_xy,16,1.0)
                    frame=overlay_rgba_on_bgr(frame,hm,alpha=heat_alpha)

            # 判定输出
            if (left_knee is not None) or (right_knee is not None):
                ang=np.nanmean([v for v in [left_knee,right_knee] if v is not None])
                if ang<=knee_thresh:
                    txt=f"{t('judge_squat')} | {t('knee_fmt').format(ang)}"; color=(0,200,0)
                else:
                    txt=f"{t('judge_not')} | {t('knee_fmt').format(ang)}"; color=(0,0,200)
                cv2.putText(frame, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            frame_box.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            interval=1.0/max(5,int(max_fps)); now=time.time(); sleep_t=interval-(now-last); last=now
            if sleep_t>0: time.sleep(sleep_t)

        cap.release(); pose.close()
else:
    st.caption(t("wait_start"))
