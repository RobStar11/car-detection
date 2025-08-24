# -*- coding: utf-8 -*-
"""
4 cámaras físicas -> mosaico 2x2 en una sola ventana
- Escanea SOLO 0,2,4,6 (tus nodos buenos)
- Fuerza MJPG 640x480 @15fps para evitar saturar USB
- Usa YOLOv5 ./best.pt
- Una ventana, ESC/q para salir
"""

import os, warnings, platform, cv2, torch, numpy as np
from time import sleep
from pathlib import Path

# ---- antes de cv2/torch ----
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# ===== Config =====
WEIGHTS = Path("./best.pt")
SCAN_RANGE = [0, 2, 4, 6]        # <--- SOLO estos índices
MAX_CAMERAS = 4
TILE_W, TILE_H = 640, 360
WIN_NAME = "YOLOv5 - best.pt (4 cams)"
INFER_SIZE = 512                 # un poco más liviano que 640
FORCE_W, FORCE_H = 640, 480      # <--- fuerza resolución ligera
FORCE_FPS = 15                   # <--- baja FPS
FORCE_MJPG = True                # <--- fuerza MJPG

if not WEIGHTS.exists():
    raise FileNotFoundError(f"No se encontró {WEIGHTS.resolve()}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("OpenCV:", cv2.__version__, "| SO:", platform.system(), platform.release())
print(f"Usando dispositivo: {DEVICE}")

# ===== Modelo =====
model = torch.hub.load(
    'ultralytics/yolov5', 'custom', path=str(WEIGHTS),
    source='github', trust_repo=True, force_reload=False
).to(DEVICE)
model.conf = 0.25
model.iou  = 0.45

def _has(name: str) -> bool:
    return hasattr(cv2, name)

def _force_params(cap):
    if FORCE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if FORCE_W:   cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FORCE_W)
    if FORCE_H:   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FORCE_H)
    if FORCE_FPS: cap.set(cv2.CAP_PROP_FPS,          FORCE_FPS)

def try_open(index, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(index)
        tag = "default"
    else:
        cap = cv2.VideoCapture(index, backend)
        tag = f"backend={backend}"

    if not cap.isOpened():
        return None, f"❌ No abre: /dev/video{index} ({tag})"

    _force_params(cap)
    ok, _ = cap.read()
    if ok:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap, f"✅ /dev/video{index} OK {w}x{h}@{fps:.0f} ({tag})"
    cap.release()
    return None, f"❌ read() falló: /dev/video{index} ({tag})"

def discover_cameras(max_cams=4):
    found = []
    backends = []
    if _has('CAP_V4L2'):      backends.append(cv2.CAP_V4L2)      # prioriza V4L2
    if _has('CAP_GSTREAMER'): backends.append(cv2.CAP_GSTREAMER) # luego GStreamer
    backends.append(None)                                         # fallback default

    for idx in SCAN_RANGE:
        for be in backends:
            cap, msg = try_open(idx, be)
            print(msg)
            if cap is not None:
                found.append((idx, cap))
                break
        if len(found) >= max_cams:
            break

    while len(found) < max_cams:
        found.append((None, None))
    return found[:max_cams]

def make_placeholder(text, w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0,0), (w-1,h-1), (80,80,80), 2)
    cv2.putText(img, text, (20, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2, cv2.LINE_AA)
    return img

def label_tile(img, text):
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return img

caps = discover_cameras(MAX_CAMERAS)

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TILE_W*2, TILE_H*2)
sleep(0.1)

def read_annotate_or_placeholder(idx, cap):
    if cap is None:
        return make_placeholder("No signal", TILE_W, TILE_H)
    ok, frame = cap.read()
    if not ok or frame is None:
        return make_placeholder(f"Cam {idx} sin señal", TILE_W, TILE_H)
    results = model(frame, size=INFER_SIZE)
    annotated = results.render()[0]
    tile = cv2.resize(annotated, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    label = f"Cam {idx}" if idx is not None else "Cam ?"
    return label_tile(tile, label)

with torch.inference_mode():
    while True:
        tiles = [read_annotate_or_placeholder(idx, cap) for idx, cap in caps]
        mosaic = np.vstack((np.hstack((tiles[0], tiles[1])),
                            np.hstack((tiles[2], tiles[3]))))
        cv2.imshow(WIN_NAME, mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break

for _, cap in caps:
    if cap is not None: cap.release()
cv2.destroyAllWindows()
