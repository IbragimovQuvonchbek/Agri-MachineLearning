import os
import json
import uuid
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from yolo_video import process_video_stream
from llm import ask_llm   # ✅ OpenAI / DeepSeek abstraction

load_dotenv()
# =========================
# App setup
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Serve detection outputs
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")

# Track latest run for /crops
LATEST_RUN = {
    "run_dir": None,
    "video_out": None,
}

# =========================
# Basic routes
# =========================
@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# =========================
# Upload video
# =========================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    vid_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_DIR, f"{vid_id}{ext}")

    with open(path, "wb") as f:
        f.write(await file.read())

    return {"video_path": path}

# =========================
# List detected crops
# =========================
@app.get("/crops")
def list_crops():
    run_dir = LATEST_RUN.get("run_dir")
    if not run_dir:
        return {"crops": []}

    crops_dir = os.path.join(run_dir, "crops")
    if not os.path.exists(crops_dir):
        return {"crops": []}

    run_name = os.path.basename(run_dir)
    files = sorted(f for f in os.listdir(crops_dir) if f.lower().endswith(".jpg"))

    urls = [f"/runs/{run_name}/crops/{f}" for f in files]
    return {"crops": urls}

# =========================
# WebSocket: live analysis
# =========================
@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    await websocket.accept()

    try:
        msg = await websocket.receive_json()
        video_path = msg.get("video_path")
        if not video_path:
            await websocket.send_text(json.dumps({"error": "video_path missing"}))
            return

        for event in process_video_stream(
            video_path=video_path,
            model_path="best.pt",
            conf=0.35,
            iou_thresh=0.5,
            imgsz=640,
            max_det=100,
        ):
            event_type = event[0]

            # 🔴 frame event
            if event_type == "frame":
                _, frame_bytes, run_dir, out_path = event
                LATEST_RUN["run_dir"] = run_dir
                LATEST_RUN["video_out"] = out_path
                await websocket.send_bytes(frame_bytes)

            # 🟢 new crop event
            elif event_type == "new_crop":
                _, crop_file, run_dir = event
                run_name = os.path.basename(run_dir)
                await websocket.send_text(json.dumps({
                    "type": "new_crop",
                    "url": f"/runs/{run_name}/crops/{crop_file}"
                }))

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass

# =========================
# LLM Chat
# =========================
class ChatRequest(BaseModel):
    crop_url: str
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    """
    LLM-based food quality & safety assistant
    """

    filename = os.path.basename(req.crop_url)
    # Example: Fresh_Potato_0.92_000012.jpg
    parts = filename.replace(".jpg", "").split("_")

    freshness = "unknown"
    vegetable = "vegetable"

    if len(parts) >= 2:
        freshness = parts[0].lower()
        vegetable = parts[1].lower()

    system_prompt = (
        "You are a food quality and safety assistant. "
        "You provide cautious, practical advice about vegetables. "
        "If a vegetable is rotten, clearly warn the user not to eat it. "
        "Do not give medical advice. Keep answers short and clear."
    )

    user_prompt = (
        f"The AI system detected a {freshness} {vegetable}.\n\n"
        f"User question: {req.question}"
    )

    try:
        answer = ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider="openai"   # ✅ change to 'deepseek' if needed
        )
        return {"answer": answer}

    except Exception as e:
        return {"answer": f"LLM error: {str(e)}"}
