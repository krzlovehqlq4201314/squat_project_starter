import argparse, sys, torch
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/best_model.pth")
ap.add_argument("--out",  default="models/squat.onnx")
ap.add_argument("--img_size", type=int, default=224)
ap.add_argument("--num_classes", type=int, default=2)  # 若 ckpt 带 classes 会覆盖
args = ap.parse_args()

def load_sd_and_classes(p):
    obj = torch.load(p, map_location="cpu")
    classes = None
    if isinstance(obj, dict):
        if "classes" in obj and isinstance(obj["classes"], (list,tuple)):
            classes = list(obj["classes"])
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"], classes
        return obj, classes
    raise ValueError("Unknown checkpoint format")

# 1) 取 state_dict & classes
sd, classes = load_sd_and_classes(args.ckpt)
if classes is not None:
    print("[INFO] classes from ckpt:", classes)
    args.num_classes = len(classes)
print("[INFO] num_classes used:", args.num_classes)

# 2) 有 src/model.py 就用 build_resnet18，否则退回 torchvision
model = None
src_dir = Path(__file__).resolve().parents[1] / "src"
if (src_dir / "model.py").exists():
    sys.path.append(str(src_dir))
    try:
        from model import build_resnet18
        model = build_resnet18(num_classes=args.num_classes, pretrained=False)
        print("[INFO] using src/model.py:build_resnet18")
    except Exception as e:
        print("[WARN] failed to import build_resnet18, fallback to torchvision:", e)

if model is None:
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    print("[INFO] using torchvision.models.resnet18 fallback")

# 3) 加载权重（宽松）
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:   print("[WARN] missing keys:", missing)
if unexpected:print("[WARN] unexpected keys:", unexpected)

# 4) 导出 ONNX
model.eval()
dummy = torch.randn(1, 3, args.img_size, args.img_size)
outp  = Path(args.out)
outp.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    model, dummy, str(outp),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18
)
print("[OK] exported:", outp.resolve())

# 5) 写出 classes.txt 供前端显示
txt = outp.parent / "classes.txt"
if classes is None:
    classes = (["correct","incorrect","not_squat"] if args.num_classes==3
               else ["not_squat","squat"])
txt.write_text("\n".join(classes), encoding="utf-8")
print("[OK] wrote classes:", txt.resolve())
