import streamlit as st
import numpy as np, cv2, pandas as pd
import mediapipe as mp

st.set_page_config(page_title="上传图片评估", layout="wide")
st.header("🖼️ 上传图片评估 — Squat Checker")

with st.sidebar:
    st.subheader("参数")
    th = st.slider("判定为 squat 的膝关节角度阈值(°)", 80, 140, 110, 1)
    judge_mode = st.selectbox("判定方式", ["取较小角度(更稳)", "取平均角度"])
    show_heat = st.checkbox("显示热力图", value=True)
    show_skel = st.checkbox("显示关键线", value=True)
    heat_alpha = st.slider("热力图透明度", 0.0, 1.0, 0.45, 0.01)

file = st.file_uploader("上传 JPG/PNG", type=["jpg","jpeg","png"])
if not file:
    st.info("👉 请上传一张包含下肢的照片")
    st.stop()

data = np.frombuffer(file.read(), np.uint8)
img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("无法解码图片")
    st.stop()
H, W = img_bgr.shape[:2]

mp_pose = mp.solutions.pose
with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)

if not res.pose_landmarks:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
    st.warning("未检测到人体关键点，请换一张清晰/完整的人像图。")
    st.stop()

lm = res.pose_landmarks.landmark
def xy(i):
    return np.array([lm[i].x*W, lm[i].y*H], dtype=np.float32)

def ok(i, vth=0.55):
    try:
        return (lm[i].visibility or 0) >= vth
    except Exception:
        return True

def angle3(A, B, C):
    v1 = A - B; v2 = C - B
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return np.nan
    v1 = v1/np.linalg.norm(v1); v2 = v2/np.linalg.norm(v2)
    cos = np.clip((v1*v2).sum(), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

L_HIP, L_KNEE, L_ANK = 23, 25, 27
R_HIP, R_KNEE, R_ANK = 24, 26, 28

l_valid = ok(L_HIP) and ok(L_KNEE) and ok(L_ANK)
r_valid = ok(R_HIP) and ok(R_KNEE) and ok(R_ANK)

l_knee = angle3(xy(L_HIP), xy(L_KNEE), xy(L_ANK)) if l_valid else np.nan
r_knee = angle3(xy(R_HIP), xy(R_KNEE), xy(R_ANK)) if r_valid else np.nan

knee_angles = [l_knee, r_knee]
avg_knee = np.nanmean(knee_angles)
min_knee = np.nanmin(knee_angles)

# 选择判定角度：默认“取较小角度”，可在侧栏切换为“平均值”
use_knee = min_knee if judge_mode.startswith("取较小") else avg_knee

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
prob_squat = float(sigmoid((th - use_knee)/8.0))
prob_not  = 1.0 - prob_squat
label = "squat" if prob_squat >= 0.5 else "not_squat"
conf = max(prob_squat, prob_not)

draw = img_bgr.copy()

if show_skel:
    pairs = [(11,13),(13,15),(12,14),(14,16),
             (11,23),(12,24),(23,25),(25,27),
             (24,26),(26,28),(23,24)]
    for a,b in pairs:
        pa, pb = xy(a).astype(int), xy(b).astype(int)
        cv2.line(draw, tuple(pa), tuple(pb), (0,255,255), 3, cv2.LINE_AA)
    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
        p = xy(i).astype(int)
        cv2.circle(draw, tuple(p), 5, (0,255,0), -1, cv2.LINE_AA)

if show_heat:
    heat = np.zeros((H, W), np.float32)
    for i in [11,12,13,14,15,16,23,24,25,26,27,28]:
        p = xy(i).astype(int)
        cv2.circle(heat, (p[0], p[1]), 14, 1.0, -1, cv2.LINE_AA)
    heat = cv2.GaussianBlur(heat, (0,0), 25)
    if heat.max() > 1e-6:
        heat = (np.clip(heat/heat.max(), 0, 1)*255).astype(np.uint8)
        heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        draw = cv2.addWeighted(draw, 1.0, heat_c, float(heat_alpha), 0)

c1, c2 = st.columns([2,1], gap="large")
with c1:
    st.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), caption="分析结果", use_container_width=True)

with c2:
    st.subheader("判定")
    st.markdown(f"**预测**：`{label}`  |  **置信度**：`{conf*100:.1f}%`")
    st.markdown(f"- 左膝角度：`{l_knee:.1f}°`{'（无效）' if np.isnan(l_knee) else ''}")
    st.markdown(f"- 右膝角度：`{r_knee:.1f}°`{'（无效）' if np.isnan(r_knee) else ''}")
    st.markdown(f"- 判定方式：`{judge_mode}`")
    st.markdown(f"- 用于判定的角度：`{use_knee:.1f}°`  |  阈值：`{th}°`")

    st.subheader("动作建议")
    tips = []
    if prob_squat < 0.5:
        tips.append("再下蹲一些（膝角度再小一点），保持髋部向后坐。")
    if not np.isnan(l_knee) and not np.isnan(r_knee) and abs(l_knee-r_knee) > 12:
        tips.append("两侧膝角差异较大，注意左右均衡发力。")
    try:
        l_vec = xy(25)[0] - xy(27)[0]
        r_vec = xy(26)[0] - xy(28)[0]
        if l_vec < -5 or r_vec < -5:
            tips.append("留意膝盖内扣，尽量保持膝盖朝向脚尖。")
    except Exception:
        pass
    if not tips:
        tips = ["动作看起来不错，保持住！"]
    for t in tips:
        st.write("• " + t)

st.subheader("概率分布")
st.bar_chart(pd.DataFrame({"prob":[prob_not, prob_squat]}, index=["not_squat","squat"]))
