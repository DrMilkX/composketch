"""
Peak Shot Explorer
──────────────────
Flask server that:
  1. Serves  /          → gallery frontend  (templates/index.html)
  2. Serves  /scene     → agent Three.js scene (templates/scene.html)
  3. Streams /api/explore via SSE:
       - Launches a Selenium-controlled Chrome window on /scene
       - Teleports a virtual camera to random positions
       - Classifies each frame with peak_model.pt (EfficientNet-B0)
       - Streams each "good" image back as a base64 JPEG + confidence score
       - Stops after 6 good images (or MAX_STEPS attempts)

Run:
    pip install flask
    python app.py
Then open http://localhost:5000
"""

import io, base64, json, math, random, threading, time
from pathlib import Path

from flask import Flask, Response, render_template

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
DRIVER_PATH  = str(BASE / "chromedriver")
MODEL_PATH   = str(BASE / "peak_model.pt")
CAPTURES_DIR = BASE / "static" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS   = 160    # max random poses to try
TARGET_GOOD = 6      # stop collecting when we hit this
AREA_RADIUS = 9.0    # stay within 9 of the 10-unit scene radius

# ── Peak model ────────────────────────────────────────────────────────────────
def _build():
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_f, 2))
    return m

_peak = _build().to(DEVICE)
_peak.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_peak.eval()
print(f"✓  peak_model loaded  ({DEVICE})")

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify(img: Image.Image) -> dict:
    t = _tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_peak(t), dim=1).squeeze()
    lbl = int(probs.argmax())
    return {"label": lbl, "prob_good": round(float(probs[1]), 4)}

# ── Flask ─────────────────────────────────────────────────────────────────────
app   = Flask(__name__)
_lock = threading.Lock()   # allow only one active exploration at a time


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scene")
def scene():
    return render_template("scene.html")


# ── SSE helper ────────────────────────────────────────────────────────────────
def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Exploration stream ────────────────────────────────────────────────────────
@app.route("/api/explore")
def explore():
    """
    Server-Sent Events stream.
    Emits JSON objects:
      {status:"loading",  msg: str}
      {status:"exploring", step:int, good:int, total:int}
      {status:"image",    index:int, b64:str, prob_good:float, step:int}
      {status:"done",     good:int, steps:int}
      {error: str}
    """
    if not _lock.acquire(blocking=False):
        def _busy():
            yield sse({"error": "Another exploration is already running — try again shortly."})
        return Response(_busy(), mimetype="text/event-stream")

    # Wipe old captures
    for f in CAPTURES_DIR.glob("*.jpg"):
        f.unlink(missing_ok=True)

    def generate():
        driver = None
        try:
            # ── Launch browser ────────────────────────────────────────────────
            opts = Options()
            opts.add_argument("--window-size=1280,720")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--disable-infobars")
            opts.add_argument("--disable-extensions")
            # ↓ uncomment to run invisible (WebGL still works in newer Chrome)
            # opts.add_argument("--headless=new")
            # opts.add_argument("--use-gl=angle")

            driver = webdriver.Chrome(
                service=Service(DRIVER_PATH),
                options=opts,
            )

            # ── Load scene ────────────────────────────────────────────────────
            yield sse({"status": "loading", "msg": "Generating 3D scene…"})
            driver.get(f"http://127.0.0.1:{PORT}/scene")

            # Wait up to 15 s for Three.js scene to finish loading
            for _ in range(150):
                if driver.execute_script("return window.sceneReady === true;"):
                    break
                time.sleep(0.1)
            else:
                yield sse({"error": "Scene timed out — Three.js CDN may be unreachable."})
                return

            time.sleep(0.6)   # let shadow maps settle
            yield sse({"status": "exploring", "step": 0, "good": 0, "total": MAX_STEPS})

            # ── Random-pose agent loop ────────────────────────────────────────
            good_count = 0
            step = 0

            while good_count < TARGET_GOOD and step < MAX_STEPS:
                step += 1

                # Sample a random pose inside the scene
                ang   = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(0.8, AREA_RADIUS)
                x     = round(math.cos(ang) * dist, 3)
                z     = round(math.sin(ang) * dist, 3)
                yaw   = round(random.uniform(0, 2 * math.pi), 3)
                # Slight downward pitch so shapes fill the frame better
                pitch = round(random.uniform(-0.28, 0.04), 3)

                driver.execute_script(
                    f"window.agentTeleport({x},{z},{yaw},{pitch});"
                )
                time.sleep(0.12)   # let the frame render

                b64 = driver.execute_script("return window.agentSnapshot();")
                if not b64 or "," not in b64:
                    continue

                # Decode → classify
                raw    = base64.b64decode(b64.split(",")[1])
                pil    = Image.open(io.BytesIO(raw)).convert("RGB")
                result = classify(pil)

                if result["label"] == 1:      # good frame
                    fname = f"cap_{good_count:02d}.jpg"
                    pil.save(CAPTURES_DIR / fname, quality=90)

                    yield sse({
                        "status":    "image",
                        "index":     good_count,
                        "b64":       b64,
                        "prob_good": result["prob_good"],
                        "step":      step,
                    })
                    good_count += 1

                # Periodic progress heartbeat
                elif step % 8 == 0:
                    yield sse({
                        "status": "exploring",
                        "step":   step,
                        "good":   good_count,
                        "total":  MAX_STEPS,
                    })

            yield sse({"status": "done", "good": good_count, "steps": step})

        except Exception as exc:
            yield sse({"error": str(exc)})

        finally:
            if driver:
                driver.quit()
            _lock.release()

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


PORT = 5055   # 5000 is reserved by macOS AirPlay Receiver

if __name__ == "__main__":
    app.run(debug=False, port=PORT, threaded=True)
