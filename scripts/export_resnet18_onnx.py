import argparse, torch, torch.nn as nn
from torchvision import models

def build_model(num_classes=2):
    # 兼容 torchvision 0.24 的API：weights=None 是合法的
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="checkpoint .pth（可空）")
    ap.add_argument("--out",  type=str, default="models/squat.onnx")
    ap.add_argument("--img",  type=int, default=224, help="导出时的输入分辨率")
    args = ap.parse_args()

    model = build_model(2)

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
            print("✓ checkpoint loaded (strict=False)")
        except Exception as e:
            print("! load_state_dict error:", e)

    model.eval()
    dummy = torch.randn(1,3,args.img,args.img)

    # torch 2.9 建议用更高的 opset（如 17），兼容性更好
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        opset_version=18, do_constant_folding=True,
        dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}
    )
    print("✓ exported:", args.out)

if __name__ == "__main__":
    main()
