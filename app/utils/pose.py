from __future__ import annotations
import math
import cv2
import numpy as np
import streamlit as st

COCO_PAIRS = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(5,11),(6,12),(11,12),(0,5),(0,6)
]

@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")

def _angle(a,b,c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    nba = np.linalg.norm(ba) + 1e-8
    nbc = np.linalg.norm(bc) + 1e-8
    cosang = float(np.dot(ba, bc) / (nba*nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def _draw_skeleton(img, kpt):
    for i, j in COCO_PAIRS:
        if i < len(kpt) and j < len(kpt):
            xi, yi = kpt[i]
            xj, yj = kpt[j]
            if not (np.isnan(xi) or np.isnan(yi) or np.isnan(xj) or np.isnan(yj)):
                cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), (0,255,0), 2, cv2.LINE_AA)
    for (x, y) in kpt:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(img, (int(x), int(y)), 3, (0,255,255), -1, cv2.LINE_AA)

def _heatmap(h, w, kpt, sigma=18):
    hm = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for (x, y) in kpt:
        if np.isnan(x) or np.isnan(y): 
            continue
        g = np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma*sigma))
        hm = np.maximum(hm, g)
    hm = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(hm, cv2.COLORMAP_JET)

def analyze(img_bgr: np.ndarray, draw_lines: bool=True, draw_heat: bool=True):
    model = load_model()
    results = model.predict(source=img_bgr, verbose=False, imgsz=640, device='cpu')
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return img_bgr, "NOT SQUAT ❌ | No person detected. Ensure whole body is in frame."

    k = results[0].keypoints.xy[0].cpu().numpy()
    if k.shape[0] < 17:
        pad = np.full((17-k.shape[0], 2), np.nan, dtype=np.float32)
        k = np.concatenate([k, pad], axis=0)

    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()

    nose   = k[0]
    lsh, rsh = k[5], k[6]
    lhip, rhip = k[11], k[12]
    lknee, rknee = k[13], k[14]
    lank,  rank  = k[15], k[16]

    sh_mid  = np.nanmean(np.vstack([lsh, rsh]), axis=0)
    hip_mid = np.nanmean(np.vstack([lhip, rhip]), axis=0)
    knee_mid= np.nanmean(np.vstack([lknee, rknee]), axis=0)
    ank_mid = np.nanmean(np.vstack([lank, rank]), axis=0)

    def _safe(p): return not (np.isnan(p[0]) or np.isnan(p[1]))
    def _side_depth_ok(hip,knee,tol): return _safe(hip) and _safe(knee) and (hip[1] > knee[1] - tol)

    angles={}
    if _safe(lhip) and _safe(lknee) and _safe(lank):
        angles["knee_L"] = _angle(lhip, lknee, lank)
    if _safe(rhip) and _safe(rknee) and _safe(rank):
        angles["knee_R"] = _angle(rhip, rknee, rank)
    knee_angle = np.nanmean([angles.get("knee_L", np.nan), angles.get("knee_R", np.nan)])

    torso_lean = np.nan
    if _safe(hip_mid) and _safe(sh_mid):
        v = sh_mid - hip_mid
        vertical = np.array([0, 1.0])
        torso_lean = _angle(hip_mid + vertical, hip_mid, sh_mid)

    tol = 0.02 * H
    cand = []
    if _safe(lhip) and _safe(lknee): cand.append(_side_depth_ok(lhip, lknee, tol))
    if _safe(rhip) and _safe(rknee): cand.append(_side_depth_ok(rhip, rknee, tol))
    if cand:
        depth_ok = any(cand)
    else:
        depth_ok = _safe(hip_mid) and _safe(knee_mid) and (hip_mid[1] > knee_mid[1] - tol)

    knee_track_ok = True
    if _safe(lknee) and _safe(lank) and _safe(rknee) and _safe(rank) and _safe(lsh) and _safe(rsh):
        shoulder_w = abs(rsh[0] - lsh[0]) + 1e-5
        dev_L = abs(lknee[0] - lank[0]) / shoulder_w
        dev_R = abs(rknee[0] - rank[0]) / shoulder_w
        knee_track_ok = (dev_L < 0.25) and (dev_R < 0.25)

    hip_angle = np.nan
    if _safe(lsh) and _safe(lhip) and _safe(lknee):
        hip_angle = _angle(lsh, lhip, lknee)
    if _safe(rsh) and _safe(rhip) and _safe(rknee):
        hip_angle = np.nanmean([hip_angle, _angle(rsh, rhip, rknee)])

    # 阈值（和你当前一致/放宽）
    ok_knee   = (not np.isnan(knee_angle)) and (80 <= knee_angle <= 120)
    ok_torso  = (not np.isnan(torso_lean)) and (torso_lean <= 28)
    ok_hipAng = (not np.isnan(hip_angle)) and (hip_angle <= 110)

    passes = [ok_knee, depth_ok, ok_torso, knee_track_ok, ok_hipAng]
    all_good = all(passes)

    # —— 先做“Squat / Not Squat”总判定 —— 
    # 定义：满足深度 + 膝角在区间 + 髋角关闭 视为 SQUAT；否则 NOT SQUAT
    is_squat = depth_ok and ok_knee and ok_hipAng
    prefix = "SQUAT ✅" if is_squat else "NOT SQUAT ❌"

    # 生成信息
    tips_good, tips_fix = [], []
    if ok_knee:   tips_good.append(f"Knee angle {int(round(knee_angle))}° ✅")
    else:         tips_fix.append("Aim knees around 90° (±30° allowed).")
    if depth_ok:  tips_good.append("Depth OK ✅")
    else:         tips_fix.append("Go deeper: hip below knee at bottom (2% tolerance).")
    if ok_torso:  tips_good.append(f"Torso lean {int(round(torso_lean))}° ✅")
    else:         tips_fix.append("Keep chest up; reduce torso lean (<28°).")
    if knee_track_ok: tips_good.append("Knees track toes ✅")
    else:             tips_fix.append("Push knees out over toes (tracking).")
    if ok_hipAng: tips_good.append("Hips closed ✅")
    else:         tips_fix.append("Close hips more on descent.")

    if draw_lines: _draw_skeleton(vis, k)
    if draw_heat:
        hm = _heatmap(*vis.shape[:2], k)
        vis = cv2.addWeighted(vis, 0.65, hm, 0.35, 0)

    if is_squat and all_good:
        msg = f"{prefix} | " + " | ".join(tips_good)
    elif is_squat:
        msg = f"{prefix} | " + " | ".join(tips_good) + ("  | Tips: " + " | ".join(tips_fix) if tips_fix else "")
    else:
        # NOT SQUAT 时优先给出关键修正
        core = " | ".join(tips_fix) if tips_fix else "Form issues detected."
        extra= " | ".join(tips_good) if tips_good else ""
        msg = f"{prefix} | {core}" + (f"  | Ref: {extra}" if extra else "")

    return vis, msg
