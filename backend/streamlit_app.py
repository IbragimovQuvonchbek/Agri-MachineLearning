import os
import time
import glob
import uuid
import hashlib
import threading
import queue
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from yolo_video import process_video_stream
from llm import ask_llm

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Veg Quality AI", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RUNS_DIR = BASE_DIR / "runs"
UPLOAD_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_PATH = str(BASE_DIR / "best.pt")


# -------------------------
# Helpers
# -------------------------
def newest_run_dir() -> str | None:
    """Return the newest runs/predict-* folder (by mtime)."""
    runs = sorted(glob.glob(str(RUNS_DIR / "predict-*")), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None


def list_crops(run_dir: str) -> list[str]:
    crops_dir = Path(run_dir) / "crops"
    if not crops_dir.exists():
        return []
    files = sorted({str(p.resolve()) for p in crops_dir.glob("*.jpg")})
    return list(files)


def pretty_crop_name(path: str) -> str:
    return Path(path).name.replace("_", " ")


def parse_crop_filename(path: str) -> tuple[str, str]:
    """
    Example: Fresh_Potato_0.92_000012.jpg -> ('fresh', 'potato')
    """
    name = Path(path).stem
    parts = name.split("_")
    freshness = "unknown"
    vegetable = "vegetable"
    if len(parts) >= 2:
        freshness = parts[0].lower()
        vegetable = parts[1].lower()
    return freshness, vegetable


def save_uploaded_video(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
    vid_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{vid_id}{ext}"
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(out_path)


def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------------
# Session state init
# -------------------------
ss_init("video_path", None)
ss_init("run_dir", None)
ss_init("selected_crop", None)

ss_init("is_running", False)
ss_init("latest_frame", None)
ss_init("last_error", None)

ss_init("worker_thread", None)
ss_init("worker_queue", queue.Queue())

# Thread-safe stop signal (DO NOT read st.session_state in worker thread)
if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = threading.Event()

# For safe unique keys in crop panel
ss_init("crops_panel_version", 0)


# -------------------------
# Worker thread
# -------------------------
def yolo_worker(video_path: str, model_path: str, out_q: "queue.Queue", stop_event: threading.Event):
    """
    Always sends messages as (kind, payload, run_dir):

      ("frame", frame_bytes, run_dir)
      ("crop",  crop_path_or_None, run_dir)
      ("done",  None, None)
      ("error", error_string, None)
    """
    try:
        for event in process_video_stream(
            video_path=video_path,
            model_path=model_path,
            conf=0.35,
            iou_thresh=0.5,
            imgsz=640,
            max_det=100,
        ):
            if stop_event.is_set():
                break

            et = event[0]

            if et == "frame":
                _, frame_bytes, run_dir, _out_path = event
                out_q.put(("frame", frame_bytes, str(Path(run_dir).resolve())))

            elif et == "new_crop":
                _, crop_file, run_dir = event
                # crop_file can be path-like; keep it as string for debug if needed
                out_q.put(("crop", str(crop_file), str(Path(run_dir).resolve())))

        out_q.put(("done", None, None))

    except Exception as e:
        out_q.put(("error", str(e), None))


# -------------------------
# Crops panel (clickable)
# -------------------------
def render_crops_panel(crops_box):
    """
    Clickable crop gallery.
    Uses versioned keys to avoid StreamlitDuplicateElementKey even if called often.
    """
    st.session_state["crops_panel_version"] += 1
    v = st.session_state["crops_panel_version"]

    crops_box.empty()

    run_dir = st.session_state.get("run_dir") or newest_run_dir()
    if not run_dir:
        crops_box.info("No crops detected yet.")
        return

    st.session_state["run_dir"] = run_dir

    crops = list_crops(run_dir) or []
    crops = sorted({str(Path(p).resolve()) for p in crops})

    if not crops:
        crops_box.info("No crops detected yet.")
        return

    with crops_box.container():
        cols = st.columns(2, gap="small")
        for i, crop_path in enumerate(crops):
            with cols[i % 2]:
                st.image(crop_path, use_container_width=True)
                label = pretty_crop_name(crop_path)

                h = hashlib.sha256(crop_path.encode("utf-8")).hexdigest()[:16]
                key = f"sel_{v}_{i}_{h}"

                if st.button(f"Select: {label}", key=key):
                    st.session_state["selected_crop"] = crop_path


# -------------------------
# UI
# -------------------------
st.title("🥦 Veg Quality AI — Streamlit")

left, right = st.columns([1.7, 1.0], gap="large")

with left:
    st.subheader("Live Analysis")

    model_path = st.text_input(
        "YOLO model path",
        value=DEFAULT_MODEL_PATH,
        help="Keep best.pt in the same directory or set an absolute path.",
    )

    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        start_btn = st.button(
            "▶️ Start Analysis",
            use_container_width=True,
            disabled=(uploaded is None or st.session_state["is_running"]),
        )
    with colB:
        stop_btn = st.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=(not st.session_state["is_running"]),
        )
    with colC:
        refresh_btn = st.button("🔄 Refresh Crops", use_container_width=True)

    status_box = st.empty()
    frame_box = st.empty()

with right:
    st.subheader("Detected Crops")
    crops_box = st.empty()
    st.divider()

    st.subheader("AI Assistant")
    selected_label = st.empty()

    question = st.text_area(
        "Ask about the selected crop",
        placeholder="Ask: Can I eat this? How should I store it? What does rotten mean?",
        height=100,
    )
    ask_btn = st.button("💬 Ask AI", use_container_width=True)
    answer_box = st.empty()


# -------------------------
# Controls
# -------------------------
if refresh_btn:
    render_crops_panel(crops_box)

if stop_btn:
    # thread-safe stop
    st.session_state["stop_event"].set()

if start_btn and uploaded is not None:
    # Validate model exists
    if not os.path.exists(model_path):
        st.session_state["last_error"] = f"Model not found: {model_path}"
    else:
        st.session_state["last_error"] = None

        # reset state
        st.session_state["selected_crop"] = None
        st.session_state["run_dir"] = None
        st.session_state["latest_frame"] = None

        st.session_state["video_path"] = save_uploaded_video(uploaded)

        # reset stop event & queue
        st.session_state["stop_event"].clear()
        st.session_state["worker_queue"] = queue.Queue()

        # start worker
        t = threading.Thread(
            target=yolo_worker,
            args=(
                st.session_state["video_path"],
                model_path,
                st.session_state["worker_queue"],
                st.session_state["stop_event"],
            ),
            daemon=True,
        )
        st.session_state["worker_thread"] = t
        st.session_state["is_running"] = True
        t.start()


# -------------------------
# Drain queue updates (defensive parsing)
# -------------------------
q = st.session_state["worker_queue"]

try:
    while True:
        msg = q.get_nowait()

        # Defensive parsing to avoid ValueError from tuple-size changes
        kind = msg[0] if len(msg) > 0 else None
        payload = msg[1] if len(msg) > 1 else None
        run_dir = msg[2] if len(msg) > 2 else None

        if kind == "frame":
            st.session_state["latest_frame"] = payload
            if run_dir:
                st.session_state["run_dir"] = run_dir

        elif kind == "crop":
            if run_dir:
                st.session_state["run_dir"] = run_dir

        elif kind == "done":
            st.session_state["is_running"] = False

        elif kind == "error":
            st.session_state["last_error"] = str(payload)
            st.session_state["is_running"] = False

except queue.Empty:
    pass


# -------------------------
# Autorefresh while running (keeps UI responsive + clickable)
# -------------------------
if st.session_state["is_running"]:
    st_autorefresh(interval=500, key="live_refresh")  # refresh every 0.5 sec


# -------------------------
# Status + frame display
# -------------------------
if st.session_state["last_error"]:
    status_box.error(st.session_state["last_error"])
elif st.session_state["is_running"]:
    status_box.info("Analyzing video… (updates every 0.5s)")
else:
    status_box.success("Ready.")

if st.session_state["latest_frame"] is not None:
    frame_box.image(st.session_state["latest_frame"], caption="Live preview", use_container_width=True)
else:
    frame_box.info("Upload a video and press Start.")


# Always render crops panel (clickable)
render_crops_panel(crops_box)


# -------------------------
# Selected crop label
# -------------------------
if st.session_state.get("selected_crop"):
    selected_label.markdown(f"**Selected:** `{Path(st.session_state['selected_crop']).name}`")
else:
    selected_label.markdown("**Selected:** `none`")


# -------------------------
# Ask LLM
# -------------------------
if ask_btn:
    if not st.session_state.get("selected_crop"):
        answer_box.warning("Select a crop first.")
    elif not question.strip():
        answer_box.warning("Type a question.")
    else:
        crop_path = st.session_state["selected_crop"]
        freshness, vegetable = parse_crop_filename(crop_path)

        system_prompt = (
            "You are a food quality and safety assistant. "
            "You provide cautious, practical advice about vegetables. "
            "If a vegetable is rotten, clearly warn the user not to eat it. "
            "Do not give medical advice. Keep answers short and clear."
        )

        user_prompt = (
            f"The AI system detected a {freshness} {vegetable}.\n\n"
            f"User question: {question.strip()}"
        )

        with st.spinner("Thinking…"):
            try:
                ans = ask_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    provider="openrouter",
                )
                answer_box.success(ans)
            except Exception as e:
                answer_box.error(f"LLM error: {e}")
