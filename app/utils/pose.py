import cv2
import numpy as np

# COCO-17 keypoint indices used by YOLOv8-pose
NOSE, LE, RE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = range(17)

EDGES = [
    (LSH, RSH), (LSH, LEL), (LEL, LWR), (RSH, REL), (REL, RWR),
    (LSH, LHIP), (RSH, RHIP), (LHIP, LKNEE), (LKNEE, LANK), (RHIP, RKNEE), (RKNEE, RANK)
]

def _have(kp, *ids):
    return all((kp[i,0] > 0 and kp[i,1] > 0) for i in ids)

def _angle(a, b, c):
    # angle at b formed by (a-b) and (c-b) in degrees
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-6
    cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def _choose_side(kp):
    left_ok  = _have(kp, LHIP, LKNEE, LANK)
    right_ok = _have(kp, RHIP, RKNEE, RANK)
    if left_ok and right_ok:
        # pick the side with more confident lower-body geometry (shorter hip-ank path → less occlusion)
        left_len  = np.linalg.norm(kp[LHIP]-kp[LANK])
        right_len = np.linalg.norm(kp[RHIP]-kp[RANK])
        return ('left',  LHIP, LKNEE, LANK) if left_len > right_len else ('right', RHIP, RKNEE, RANK)
    if left_ok:  return ('left',  LHIP, LKNEE, LANK)
    if right_ok: return ('right', RHIP, RKNEE, RANK)
    return (None, None, None, None)

def _draw_skeleton(img, kp, draw_lines=True, draw_heat=False):
    canvas = img.copy()

    if draw_heat:
        heat = np.zeros_like(img, dtype=np.uint8)
        for i in range(17):
            x, y = int(kp[i,0]), int(kp[i,1])
            if x>0 and y>0:
                cv2.circle(heat, (x,y), 18, (0,255,255), -1)
        canvas = cv2.addWeighted(canvas, 1.0, heat, 0.35, 0)

    if draw_lines:
        for a,b in EDGES:
            xa,ya = int(kp[a,0]), int(kp[a,1])
            xb,yb = int(kp[b,0]), int(kp[b,1])
            if xa>0 and ya>0 and xb>0 and yb>0:
                cv2.line(canvas, (xa,ya), (xb,yb), (0,255,0), 3)

    # keypoints
    for i in range(17):
        x, y = int(kp[i,0]), int(kp[i,1])
        if x>0 and y>0:
            cv2.circle(canvas, (x,y), 5, (0,255,255), -1)

    return canvas

def analyze(model, img_bgr, draw_lines=True, draw_heat=False, knee_ok_max=95):
    """
    Returns: vis_bgr, msg (english)
    """
    # run pose
    res = model.predict(img_bgr, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.xy) == 0:
        return img_bgr, "NOT SQUAT | No person/pose detected. Please include *full body* in the frame."

    # pick the best person (highest box confidence if available)
    idx = 0
    if res.boxes is not None and len(res.boxes) > 0:
        idx = int(np.argmax(res.boxes.conf.cpu().numpy()))
    kps = res.keypoints.xy[idx].cpu().numpy()   # (17,2)

    # choose side with lower-body present
    side, hip, knee, ank = _choose_side(kps)
    if side is None:
        vis = _draw_skeleton(img_bgr, kps, draw_lines=False, draw_heat=True)
        return vis, "NOT SQUAT | Lower body not visible. Please ensure knees & ankles are inside the frame."

    # measurements (image y grows downward)
    knee_angle = float(_angle(kps[hip], kps[knee], kps[ank]))  # hip-knee-ankle
    depth_ok   = (kps[hip,1] > kps[knee,1] + 6)               # hip lower than knee
    # torso control: shoulder-hip alignment vs vertical
    sh = LSH if side=='left' else RSH
    torso_vec = kps[hip] - kps[sh]
    vertical  = np.array([0.0, 1.0])
    torso_tilt = float(np.degrees(np.arccos(np.clip(np.dot(torso_vec,vertical) /
                          (np.linalg.norm(torso_vec)*np.linalg.norm(vertical)+1e-6), -1,1))))
    torso_ok  = (torso_tilt <= 30)
    # knees track toes: knee above ankle in x within tolerance
    thigh_len = np.linalg.norm(kps[hip]-kps[knee]) + 1e-6
    knees_track = (abs(kps[knee,0]-kps[ank,0]) <= 0.18 * thigh_len)

    squat = (knee_angle <= knee_ok_max) and depth_ok

    vis = _draw_skeleton(img_bgr, kps, draw_lines=draw_lines, draw_heat=draw_heat)

    tick = lambda x: "✓" if x else "✗"
    parts = []
    parts.append(("SQUAT" if squat else "NOT SQUAT"))
    parts.append(f"Knee angle {int(round(knee_angle))}° {tick(knee_angle<=knee_ok_max)}")
    parts.append(f"Depth {tick(depth_ok)}")
    parts.append(f"Torso control {tick(torso_ok)}")
    parts.append(f"Knees track toes {tick(knees_track)}")

    return vis, " | ".join(parts)
