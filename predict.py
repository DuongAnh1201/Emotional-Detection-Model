import argparse
import sys

import cv2

from FaceDetection import FaceDetection


def main() -> int:
    p = argparse.ArgumentParser(description="Predict emotion for each face in an image file.")
    p.add_argument("image", help="path to an image file")
    p.add_argument("--conf", type=float, default=0.5, help="face detection confidence threshold")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: cannot read image '{args.image}'", file=sys.stderr)
        return 2

    fd = FaceDetection()
    res = fd.face_model.predict(img, imgsz=320, conf=args.conf, verbose=False)
    boxes = res[0].boxes.xyxy.cpu().numpy() if res and res[0].boxes is not None else []

    h, w = img.shape[:2]
    n = 0
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        feats = fd._extract_vit_features_from_crop(img[y1:y2, x1:x2])
        _, label, conf = fd._predict_emotion(feats)
        n += 1
        print(f"face {n}: {label} ({conf * 100:.1f}%) box=[{x1},{y1},{x2},{y2}]")

    if n == 0:
        print("No face detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())