import cv2
import numpy as np

POSE_PAIRS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

def _valid_xy(p):
    if p is None: return False
    x, y = p
    if x is None or y is None: return False
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return False
    if isinstance(y, float) and (np.isnan(y) or np.isinf(y)): return False
    return True


# ================== 关键修复：确保传入 cv2.line 的数组是 C 连续的 ==================

# --- patched overlay (last definition wins) ---
import cv2, numpy as np
def overlay_rgba_on_bgr(bgr, rgba, alpha: float = 1.0):
    """Overlay an RGBA/RGB image on top of a BGR frame with alpha in [0,1]."""
    if bgr is None or rgba is None:
        return bgr
    # 尺寸对齐
    if rgba.shape[:2] != bgr.shape[:2]:
        rgba = cv2.resize(rgba, (bgr.shape[1], bgr.shape[0]))
    # 拆分通道
    if rgba.shape[2] == 4:
        rgb = rgba[..., :3]
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        a = a * float(alpha)
    else:
        rgb = rgba
        a = np.full(bgr.shape[:2] + (1,), float(alpha), dtype=np.float32)
    # 颜色空间与混合
    bgr_f = bgr.astype(np.float32)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    out = (1.0 - a) * bgr_f + a * rgb_bgr
    return out.clip(0, 255).astype(np.uint8)
# --- end patch ---
# --- appended safe utils (idempotent) ---
import numpy as _np, cv2 as _cv2

def overlay_rgba_on_bgr(_bgr, _rgba, alpha=0.45):
    if _rgba is None: 
        return _bgr
    _b = _bgr.astype(_np.float32)
    if _rgba.shape[-1] == 4:
        _rgb = _rgba[...,:3].astype(_np.float32)
        _a   = (_rgba[...,3:4].astype(_np.float32)/255.0) * float(alpha)
        _out = _b*(1.0-_a) + _rgb*_a
    else:
        _rgb = _rgba.astype(_np.float32)
        _a   = float(alpha)
        _out = _b*(1.0-_a) + _rgb*_a
    return _np.clip(_out, 0, 255).astype(_np.uint8)


# --- end append ---


# ---------------- 安全工具函数 ----------------
def _safe_points(pts, w, h):
    """把关键点标准化为 [(x,y)|None,...]，并过滤越界/非法值。"""
    out = []
    for p in list(pts):
        if p is None:
            out.append(None); continue
        try:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                out.append((x, y))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out

# ---------------- 画骨架（RGBA） ----------------
def draw_skeleton_rgba(bgr, pts, thickness=2, color=(0, 255, 255)):
    import numpy as _np, cv2 as _cv2
    h, w = bgr.shape[:2]
    rgba = _np.zeros((h, w, 4), _np.uint8)

    pts = _safe_points(pts, w, h)
    # 允许动态导入 POSE_PAIRS（函数可能在本文件外定义）
    try:
        pairs = POSE_PAIRS
    except NameError:
        pairs = [
            (11,13),(13,15),(12,14),(14,16),
            (11,12),(23,24),
            (23,25),(25,27),(29,31),
            (24,26),(26,28),(30,32)
        ]

    for a, b in pairs:
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        _cv2.line(rgba, pa, pb, (int(color[0]), int(color[1]), int(color[2]), 255),
                  thickness, _cv2.LINE_AA)
        _cv2.circle(rgba, pa, 3, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
        _cv2.circle(rgba, pb, 3, (int(color[0]), int(color[1]), int(color[2]), 255), -1)
    return rgba

# ---------------- 热力图（BGR） ----------------
def build_heatmap(img_shape, pts, sigma=14, intensity=1.0):
    """根据关键点生成灰度热力图（float32, 0~1），容错 None/越界。"""
    import numpy as _np, cv2 as _cv2
    H, W = img_shape[:2]
    heat = _np.zeros((H, W), _np.float32)
    pts = _safe_points(pts, W, H)
    rad = max(3, int(sigma))
    # 用圆盘近似高斯，简单高效且稳定
    for p in pts:
        if p is None:
            continue
        _cv2.circle(heat, p, rad, float(intensity), -1, lineType=_cv2.LINE_AA)
    heat = _np.clip(heat, 0.0, 1.0)
    return heat

def colorize_heatmap(heat):
    import numpy as _np, cv2 as _cv2
    h8 = _np.uint8(_np.clip(heat, 0, 1) * 255.0)
    c = _cv2.applyColorMap(h8, _cv2.COLORMAP_JET)
    return c

