import os
import cv2
from datetime import datetime
from ultralytics import YOLO

# =========================
# Configuration
# =========================
OUT_W, OUT_H = 800, 600          # output size for website/video
THUMB_W, THUMB_H = 160, 160      # crop thumbnail size
IOU_SAME_OBJECT = 0.5            # IoU threshold for same object

# =========================
# IoU helper
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (areaA + areaB - interArea + 1e-6)

# =========================
# Video processing generator
# =========================
def process_video_stream(
    video_path: str,
    model_path: str,
    conf: float = 0.35,
    iou_thresh: float = 0.5,
    imgsz: int = 640,
    max_det: int = 100,
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_src if fps_src and fps_src > 0 else 25.0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"predict-{ts}")
    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    out_path = os.path.join(run_dir, "predicted.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    saved_objects = []  # [{"label": str, "box": (x1,y1,x2,y2)}]
    crop_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference on ORIGINAL frame
        result = model.predict(
            source=frame,
            conf=conf,
            iou=iou_thresh,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False
        )[0]

        annotated = result.plot()

        # Save crops (object-level uniqueness)
        if result.boxes is not None and len(result.boxes) > 0:
            h, w = frame.shape[:2]
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            clss  = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c, cf in zip(boxes, clss, confs):
                label = model.names.get(int(c), str(int(c))).replace(" ", "_")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                current_box = (x1, y1, x2, y2)

                # Check if same object already saved
                is_same = False
                for obj in saved_objects:
                    if obj["label"] == label and compute_iou(current_box, obj["box"]) > IOU_SAME_OBJECT:
                        is_same = True
                        break

                if is_same:
                    continue

                crop = frame[y1:y2, x1:x2]
                thumb = cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

                crop_file = f"{label}_{cf:.2f}_{crop_index:06d}.jpg"
                cv2.imwrite(os.path.join(crops_dir, crop_file), thumb)

                saved_objects.append({"label": label, "box": current_box})
                crop_index += 1

                # 🔔 notify frontend about new crop
                yield ("new_crop", crop_file, run_dir)

        # Resize annotated frame for website
        annotated_out = cv2.resize(annotated, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        writer.write(annotated_out)

        ok, jpg = cv2.imencode(".jpg", annotated_out, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            yield ("frame", jpg.tobytes(), run_dir, out_path)

    cap.release()
    writer.release()
