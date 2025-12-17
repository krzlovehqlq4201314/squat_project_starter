import numpy as np
import cv2

# 关键点可视化的骨架连线（YOLOv8-Pose / COCO-17）
COCO_PAIRS = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6),(5,11),(6,12),(11,12),
    (0,5),(0,6)
]
COCO = {
    "nose":0,"l_eye":1,"r_eye":2,"l_ear":3,"r_ear":4,
    "l_sho":5,"r_sho":6,"l_elb":7,"r_elb":8,"l_wri":9,"r_wri":10,
    "l_hip":11,"r_hip":12,"l_knee":13,"r_knee":14,"l_ank":15,"r_ank":16
}

# 阈值（可按需要微调）
CONF_THR = 0.35              # 关键点置信度阈值
FULLBODY_MIN_HEIGHT = 0.55   # 检测框高度/画面高度，保证尽量拍到全身
KNEE_THR = 105.0             # <= 此角度 + 深度达标 => SQUAT

def _get_xyc(kpt, idx):
    """返回 (x,y,conf)，若缺失或低置信度 -> (nan,nan,0)"""
    try:
        x, y = kpt.xy[0, idx].tolist()
        c = float(kpt.conf[0, idx].item())
    except Exception:
        return np.nan, np.nan, 0.0
    if np.isnan(x) or np.isnan(y) or c < CONF_THR:
        return np.nan, np.nan, 0.0
    return float(x), float(y), c

def _angle(a,b,c):
    """角ABC（度）。若任一点缺失则返回 nan"""
    ax, ay, _ = a; bx, by, _ = b; cx, cy, _ = c
    if any(np.isnan(v) for v in [ax,ay,bx,by,cx,cy]):
        return np.nan
    v1 = np.array([ax-bx, ay-by], dtype=float)
    v2 = np.array([cx-bx, cy-by], dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1==0 or n2==0: return np.nan
    cosang = np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _draw_overlay(img, kpt, draw_lines=True, draw_heat=True):
    """简单骨架+热力点可视化"""
    vis = img.copy()
    # 热力点
    if draw_heat:
        heat = np.zeros_like(vis)
        for i in range(17):
            try:
                x, y = kpt.xy[0, i].astype(int).tolist()
                c = float(kpt.conf[0, i].item())
            except Exception:
                continue
            if c < CONF_THR: 
                continue
            cv2.circle(heat, (x,y), 10, (0,0,255), -1)
        heat = cv2.GaussianBlur(heat, (0,0), 7)
        vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0.0)
    # 骨架线
    if draw_lines:
        for i,j in COCO_PAIRS:
            try:
                xi, yi = kpt.xy[0, i].astype(int).tolist()
                xj, yj = kpt.xy[0, j].astype(int).tolist()
                ci = float(kpt.conf[0, i].item()); cj = float(kpt.conf[0, j].item())
            except Exception:
                continue
            if ci>=CONF_THR and cj>=CONF_THR:
                cv2.line(vis, (xi,yi), (xj,yj), (0,255,0), 2, cv2.LINE_AA)
        for i in range(17):
            try:
                x, y = kpt.xy[0, i].astype(int).tolist()
                c = float(kpt.conf[0, i].item())
            except Exception:
                continue
            if c>=CONF_THR:
                cv2.circle(vis, (x,y), 4, (0,255,255), -1, cv2.LINE_AA)
    return vis

def analyze(image_bgr, model, draw_lines=True, draw_heat=True):
    """
    统一给“上传图片”与“Webcam Stable”调用。
    缺腿部关键点/不全身 -> 直接 NOT SQUAT + 提示。
    满足条件时计算膝角与深度再判定。
    """
    h, w = image_bgr.shape[:2]
    results = model.predict(source=image_bgr, verbose=False, imgsz=640, device="cpu")
    if not results or len(results[0].keypoints)==0:
        return image_bgr, "[NOT SQUAT] No person detected. Please include the full body."

    kpt = results[0].keypoints   # (1,17,2) + conf
    # 全身入镜：用第一个检测框判断高度占比
    fullbody_ok = True
    try:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1,y1,x2,y2 = box
        fullbody_ok = ((y2 - y1) / h) >= FULLBODY_MIN_HEIGHT
    except Exception:
        pass

    # 取左右髋/膝/踝
    l_hip = _get_xyc(kpt, COCO["l_hip"])
    r_hip = _get_xyc(kpt, COCO["r_hip"])
    l_knee = _get_xyc(kpt, COCO["l_knee"])
    r_knee = _get_xyc(kpt, COCO["r_knee"])
    l_ank = _get_xyc(kpt, COCO["l_ank"])
    r_ank = _get_xyc(kpt, COCO["r_ank"])

    left_ok  = l_hip[2]>=CONF_THR and l_knee[2]>=CONF_THR and l_ank[2]>=CONF_THR
    right_ok = r_hip[2]>=CONF_THR and r_knee[2]>=CONF_THR and r_ank[2]>=CONF_THR
    have_legs = left_ok or right_ok

    vis = _draw_overlay(image_bgr, kpt, draw_lines, draw_heat)

    # 关键点不全/非全身：直接 NOT SQUAT
    if (not have_legs) or (not fullbody_ok):
        tips = "Please show hips/knees/ankles (full body). Move back a little."
        return vis, f"[NOT SQUAT] Missing lower-body keypoints. {tips}"

    # 计算膝角（左右取均值）
    angL = _angle(l_hip, l_knee, l_ank) if left_ok else np.nan
    angR = _angle(r_hip, r_knee, r_ank) if right_ok else np.nan
    valid = [a for a in [angL, angR] if not np.isnan(a)]
    knee_angle = float(np.mean(valid)) if valid else np.nan

    # 深度：髋在膝之下视为达标（y向下为正）
    depth_ok = False
    if left_ok and not np.isnan(l_hip[1]) and not np.isnan(l_knee[1]):
        depth_ok |= (l_hip[1] > l_knee[1] * 0.98)
    if right_ok and not np.isnan(r_hip[1]) and not np.isnan(r_knee[1]):
        depth_ok |= (r_hip[1] > r_knee[1] * 0.98)

    valid_angle = not np.isnan(knee_angle)
    is_squat = valid_angle and (knee_angle <= KNEE_THR) and depth_ok

    status = "SQUAT ✅" if is_squat else "NOT SQUAT ❌"
    angle_text = f"Knee angle {knee_angle:.0f}°" if valid_angle else "Knee angle N/A"
    depth_text = "Depth OK" if depth_ok else "Depth too shallow"
    msg = f"{status} | {angle_text} | {depth_text}"

    tips = []
    if not depth_ok:
        tips.append("Go deeper (hip below knee at bottom).")
    if valid_angle and knee_angle > KNEE_THR:
        tips.append("Bend knees more (reduce knee angle).")
    if tips: msg += " | Tips: " + " ".join(tips)
    return vis, msg
