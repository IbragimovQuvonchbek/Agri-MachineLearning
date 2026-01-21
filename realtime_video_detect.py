import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = "videos/video1.mp4"

CONF = 0.35
IOU = 0.5
IMGSZ = 640
MAX_DET = 100

# ✅ output resolution only (for display + saved video)
OUT_W = 800
OUT_H = 600

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_src if fps_src and fps_src > 0 else 25.0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("runs", f"predict-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predicted.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter (check opencv-python installation).")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ YOLO on original frame (no resize)
        r = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False
        )[0]

        annotated = r.plot()

        # ✅ resize ONLY annotated output for display/save
        annotated_out = cv2.resize(annotated, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

        now = time.time()
        fps_proc = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            annotated_out,
            f"FPS: {fps_proc:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Fresh vs Rotten Vegetables (Press Q to quit)", annotated_out)
        writer.write(annotated_out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()
