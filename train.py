import argparse
from pathlib import Path
from ultralytics import YOLO

DATA    = "damages.yaml"
MODEL   = "yolov9s.pt"
PROJECT = "runs/train"
NAME    = "yolov9s_damages"
CLASS_NAMES = ["Scratched", "Dented", "Broken"] 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",   type=str,   default="0")
    p.add_argument("--patience", type=int,   default=20)
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--weights",  type=str,   default=None)
    p.add_argument("--cache",  type=str,   default=True)
    p.add_argument("--amp",  type=str,   default=True)
    p.add_argument("--seed",  type=int,   default=0)
    return p.parse_args()

def print_metrics(r, split: str):
    """Prints global mAP and per-class Precision / Recall / F1"""
    box = r.box
 
    print(f"\n{'='*54}")
    print(f"  Evaluation: {split.upper()} Dataset")
    print(f"{'='*54}")
    print(f"  mAP@50:    {box.map50:.4f}")
    print(f"  mAP@50-95: {box.map:.4f}")
 
    # Metrics per class
    # box.p, box.r → arrays de shape [n_classes]
    precision = box.p.tolist()
    recall    = box.r.tolist()
 
    # F1
    f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precision, recall)
    ]
 
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*52}")
    for i, cls in enumerate(CLASS_NAMES):
        p  = precision[i] if i < len(precision) else float("nan")
        r  = recall[i]    if i < len(recall)    else float("nan")
        f  = f1[i]        if i < len(f1)        else float("nan")
        print(f"  {cls:<15} {p:>10.4f} {r:>8.4f} {f:>8.4f}")
 
    # Macro (simple average across classes)
    valid = [(p, r, f) for p, r, f in zip(precision, recall, f1)]
    if valid:
        mp  = sum(x[0] for x in valid) / len(valid)
        mr  = sum(x[1] for x in valid) / len(valid)
        mf  = sum(x[2] for x in valid) / len(valid)
        print(f"  {'-'*52}")
        print(f"  {'Macro':<15} {mp:>10.4f} {mr:>8.4f} {mf:>8.4f}")

def main():
    args = parse_args()

    if args.resume:
        ckpt = args.weights or f"{PROJECT}/{NAME}/weights/last.pt"
        model = YOLO(ckpt)
    else:
        model = YOLO(MODEL)

    print(model.info())

    results = model.train(
        data     = DATA,
        project  = PROJECT,
        name     = NAME,
        epochs   = args.epochs,
        batch    = args.batch,
        imgsz    = args.imgsz,
        device   = args.device,
        patience = args.patience,
        resume   = args.resume,
        amp      = args.amp,
        cache    = args.cache, 
        seed     = args.seed
    )

    # Test with best weights
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        model_best = YOLO(str(best))
        r = model_best.val(
            data=DATA, 
            split="test", 
            imgsz=args.imgsz, 
            device=args.device, 
            plots=False
        )
        print_metrics(r, "test")


if __name__ == "__main__":
    main()