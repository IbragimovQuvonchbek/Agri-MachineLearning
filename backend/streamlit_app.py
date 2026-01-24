import os
import time
import glob
import uuid
from pathlib import Path
import hashlib
import streamlit as st

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
    Your crop filename format (from yolo_video.py):
      {label}_{conf:.2f}_{idx:06d}.jpg
    Example:
      Fresh_Potato_0.92_000012.jpg

    We try to infer 'freshness' + 'vegetable' from first two parts.
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


# -------------------------
# Session state
# -------------------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = 0.0


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
        start_btn = st.button("▶️ Start Analysis", use_container_width=True, disabled=(uploaded is None))
    with colB:
        stop_btn = st.button("⏹ Stop", use_container_width=True, disabled=(not st.session_state.is_running))
    with colC:
        refresh_btn = st.button("🔄 Refresh Crops", use_container_width=True)

    status = st.info("Ready.")
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
# Actions
# -------------------------
def render_crops_panel():
    """
    Renders the 'Detected Crops' panel safely (no duplicate keys), even if this
    function is called many times in the same Streamlit run.

    Assumes these exist in your file (as in your app):
      - crops_box (st.empty() placeholder)
      - newest_run_dir()
      - list_crops(run_dir)
      - pretty_crop_name(path)
      - st.session_state.run_dir
    """
    # --- Safety init (prevents AttributeError) ---
    if "crops_panel_version" not in st.session_state:
        st.session_state["crops_panel_version"] = 0

    # Bump version every render so widget keys never collide across re-renders
    st.session_state["crops_panel_version"] += 1
    v = st.session_state["crops_panel_version"]

    # Clear the placeholder before rebuilding UI
    crops_box.empty()

    # Choose run dir
    run_dir = st.session_state.get("run_dir") or newest_run_dir()
    if not run_dir:
        crops_box.info("No crops detected yet.")
        return

    st.session_state["run_dir"] = run_dir

    # List crops and dedupe by resolved absolute path (prevents duplicates)
    crops = list_crops(run_dir) or []
    crops = sorted({str(Path(p).resolve()) for p in crops})

    if not crops:
        crops_box.info("No crops detected yet.")
        return

    # Render gallery
    with crops_box.container():
        cols = st.columns(2, gap="small")
        for i, crop_path in enumerate(crops):
            with cols[i % 2]:
                st.image(crop_path, use_container_width=True)
                label = pretty_crop_name(crop_path)

                # Unique button key (version + index + stable hash)
                h = hashlib.sha256(crop_path.encode("utf-8")).hexdigest()[:16]
                key = f"sel_{v}_{i}_{h}"

                if st.button(f"Select: {label}", key=key):
                    st.session_state["selected_crop"] = crop_path


def run_analysis(video_path: str, model_path: str):
    """
    Runs your generator (process_video_stream) and streams frames into Streamlit.
    Also updates run_dir and crops panel as new crops appear.
    """
    st.session_state.is_running = True
    st.session_state.run_dir = None
    st.session_state.selected_crop = None

    # Containers updates
    status.info("Analyzing video…")
    frame_box.image(
        "https://placehold.co/720x405/e5e7eb/111827?text=Live+Preview",
        caption="Live preview",
        use_container_width=True,
    )

    last_crop_refresh = 0.0

    try:
        for event in process_video_stream(
            video_path=video_path,
            model_path=model_path,
            conf=0.35,
            iou_thresh=0.5,
            imgsz=640,
            max_det=100,
        ):
            if not st.session_state.is_running:
                status.warning("Stopped.")
                break

            event_type = event[0]

            if event_type == "frame":
                _, frame_bytes, run_dir, out_path = event
                st.session_state.run_dir = str(Path(run_dir).resolve())
                # Show frame
                frame_box.image(frame_bytes, caption="Live preview", use_container_width=True)

                # Throttle crop refresh a bit (avoid excessive UI updates)
                now = time.time()
                if now - last_crop_refresh > 1.5:
                    render_crops_panel()
                    last_crop_refresh = now

            elif event_type == "new_crop":
                # new_crop event yields: ("new_crop", crop_file, run_dir)
                _, crop_file, run_dir = event
                st.session_state.run_dir = str(Path(run_dir).resolve())
                render_crops_panel()

        status.success("Analysis finished.")
        render_crops_panel()

    except Exception as e:
        status.error(f"Error: {e}")

    finally:
        st.session_state.is_running = False


# Refresh crops manually
if refresh_btn:
    render_crops_panel()

# Stop
if stop_btn:
    st.session_state.is_running = False

# Start
if start_btn and uploaded is not None:
    st.session_state.video_path = save_uploaded_video(uploaded)
    # Ensure model exists
    if not os.path.exists(model_path):
        status.error(f"Model not found: {model_path}")
    else:
        run_analysis(st.session_state.video_path, model_path)

# Always render crops panel once (if any)
if not st.session_state.is_running:
    render_crops_panel()

# Selected crop label
if st.session_state.selected_crop:
    selected_label.markdown(f"**Selected:** `{Path(st.session_state.selected_crop).name}`")
else:
    selected_label.markdown("**Selected:** `none`")

# Ask LLM
if ask_btn:
    if not st.session_state.selected_crop:
        answer_box.warning("Select a crop first.")
    elif not question.strip():
        answer_box.warning("Type a question.")
    else:
        crop_path = st.session_state.selected_crop
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
