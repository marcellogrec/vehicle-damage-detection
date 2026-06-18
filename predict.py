import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ["Scratched", "Dented", "Broken"]
COLORS = {
    "Scratched": (0, 255, 0),    # green
    "Dented":    (0, 165, 255),  # orange
    "Broken":    (0, 0, 255),    # red
}

def parse_args():
    p = argparse.ArgumentParser(description="Vehicle damage detection inference")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to trained model weights (runs/detect/runs/train/yolo12s_damages/weights/best.pt)")
    p.add_argument("--image",   type=str, required=True,
                   help="Path to input image")
    p.add_argument("--imgsz",   type=int, default=640,
                   help="Inference image size (default: 640)")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="IoU threshold for NMS (default: 0.45)")
    p.add_argument("--device",  type=str, default="0",
                   help="Device to use: '0' for GPU, 'cpu' for CPU")
    p.add_argument("--output",  type=str, default=None,
                   help="Path to save output image (optional)")
    return p.parse_args()


def draw_predictions(image: np.ndarray, result) -> np.ndarray:
    img = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found")
        return img

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        color    = COLORS.get(cls_name, (255, 255, 255))
        label    = f"{cls_name} {conf:.2f}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # Label Background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(img, (x1, top), (x1 + tw + 4, y1), color, thickness=-1)

        # Label text
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=2,
                    lineType=cv2.LINE_AA)

    return img


def print_detections(result):
    print(f"\n{'='*50}")
    print(f"Detections")
    print(f"{'='*50}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found")
        return

    print(f"{'#':<4} {'Class':<12} {'Confidence':>10}  {'Bounding Box (x1,y1,x2,y2)'}")
    print(f"{'-'*58}")

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        coords = [int(v) for v in box.xyxy[0].tolist()]
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  {i+1:<4} {cls_name:<12} {conf:>10.4f}  {coords}")

    print(f"\nTotal: {len(result.boxes)} detections")


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    image_path   = Path(args.image)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load model
    print(f"\nLoading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Inference
    print(f"Processing image: {image_path}")
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False
    )

    result = results[0]

    # show detections in terminal
    print_detections(result)

    # draw bounding boxes
    image = cv2.imread(str(image_path))
    output_img = draw_predictions(image, result)

    # save or show imagem
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), output_img)
        print(f"\nImage saved in: {out_path}")
    else:
        out_path = image_path.parent / f"{image_path.stem}_pred{image_path.suffix}"
        cv2.imwrite(str(out_path), output_img)
        print(f"\nImage saved in: {out_path}")

    # show window
    try:
        cv2.imshow("Vehicle Damage Detection", output_img)
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass 

if __name__ == "__main__":
    main()
