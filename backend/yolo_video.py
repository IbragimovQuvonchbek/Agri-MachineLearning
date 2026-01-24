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

# Performance knobs (FAST)
DEFAULT_FRAME_SKIP = 2           # process 1 out of (skip+1) frames. 2 => every 3rd frame
DEFAULT_RENDER_FPS = 8           # how many frames/sec to yield to UI (lower = faster UI)
DEFAULT_INFER_SCALE = 0.75       # downscale factor for inference (0.5-1.0). Lower = faster

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
    imgsz: int = 416,              # faster default than 640
    max_det: int = 20,             # faster default than 100
    frame_skip: int = DEFAULT_FRAME_SKIP,
    render_fps: int = DEFAULT_RENDER_FPS,
    infer_scale: float = DEFAULT_INFER_SCALE,
    jpeg_quality: int = 70,
):
    """
    Yields:
      ("new_crop", crop_file, run_dir)
      ("frame", jpg_bytes, run_dir, out_path)

    Speed improvements:
      - frame_skip: skip frames before YOLO
      - infer_scale: downscale frame for YOLO inference
      - render_fps: only yield frames at most N times/sec
    """
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

    frame_idx = 0
    last_annotated_out = None

    # UI yield limiter (time-based)
    min_yield_interval = 1.0 / max(1, int(render_fps))
    last_yield_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed
        if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
            # Still write the last annotated frame to output video to keep length similar
            if last_annotated_out is not None:
                writer.write(last_annotated_out)
            continue

        h0, w0 = frame.shape[:2]

        # Downscale for inference (much faster)
        if infer_scale and infer_scale != 1.0:
            inf_w = max(64, int(w0 * infer_scale))
            inf_h = max(64, int(h0 * infer_scale))
            frame_inf = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
            scale_x = w0 / inf_w
            scale_y = h0 / inf_h
        else:
            frame_inf = frame
            scale_x = 1.0
            scale_y = 1.0

        # YOLO inference
        result = model.predict(
            source=frame_inf,
            conf=conf,
            iou=iou_thresh,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False
        )[0]

        annotated_inf = result.plot()

        # Convert boxes back to ORIGINAL frame coords for cropping
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            clss  = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c, cf in zip(boxes, clss, confs):
                label = model.names.get(int(c), str(int(c))).replace(" ", "_")

                # Scale boxes to original frame coords
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                current_box = (x1, y1, x2, y2)

                # uniqueness check
                is_same = False
                for obj in saved_objects:
                    if obj["label"] == label and compute_iou(current_box, obj["box"]) > IOU_SAME_OBJECT:
                        is_same = True
                        break
                if is_same:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                thumb = cv2.resize(crop, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                crop_file = f"{label}_{cf:.2f}_{crop_index:06d}.jpg"
                cv2.imwrite(os.path.join(crops_dir, crop_file), thumb)

                saved_objects.append({"label": label, "box": current_box})
                crop_index += 1

                yield ("new_crop", crop_file, run_dir)

        # Resize annotated for output/UI
        annotated_out = cv2.resize(annotated_inf, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        last_annotated_out = annotated_out
        writer.write(annotated_out)

        # Yield frames at limited FPS to keep Streamlit responsive
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if (now - last_yield_time) >= min_yield_interval:
            ok, jpg = cv2.imencode(".jpg", annotated_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if ok:
                yield ("frame", jpg.tobytes(), run_dir, out_path)
                last_yield_time = now

    cap.release()
    writer.release()
