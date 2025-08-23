# -*- coding: utf-8 -*-
"""
4 cámaras físicas -> mosaico 2x2 en una sola ventana
- Auto-descubre hasta 4 cámaras (escanea 0..9)
- Ubuntu: prioriza V4L2, luego GStreamer, luego default
- Usa tu modelo YOLOv5 custom ./best.pt
- Limpio, estable (1 ventana), ESC/q para salir
"""

# ---- Antes de importar cv2/torch ----
import os, warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"  # imshow más estable en X11
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# ------------------------------------------------------------------

import platform
import cv2
import torch
import numpy as np
from time import sleep
from pathlib import Path

# ===== Config =====
WEIGHTS = Path("./best.pt")       # tu modelo entrenado en la raíz
SCAN_RANGE = range(0, 10)         # índices que se intentan abrir (0..9)
MAX_CAMERAS = 4                   # cuántas cámaras queremos mostrar
TILE_W, TILE_H = 640, 360         # tamaño de cada celda en el mosaico
WIN_NAME = "YOLOv5 - best.pt (4 cams)"
INFER_SIZE = 640                  # tamaño de entrada YOLO
FORCE_W, FORCE_H = None, None     # p.ej. 1280, 720 si quieres forzar
FORCE_FPS = None                  # p.ej. 30

# ===== Chequeos =====
if not WEIGHTS.exists():
    raise FileNotFoundError(f"No se encontró {WEIGHTS.resolve()} — ajusta WEIGHTS o coloca best.pt en la raíz.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("OpenCV:", cv2.__version__, "| SO:", platform.system(), platform.release())
print(f"Usando dispositivo: {DEVICE}")

# ===== Modelo =====
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=str(WEIGHTS),
    source='github',
    trust_repo=True,
    force_reload=False
).to(DEVICE)
model.conf = 0.25
model.iou  = 0.45

def _has(name: str) -> bool:
    return hasattr(cv2, name)

def try_open(index, backend=None):
    """Intenta abrir cámara índice 'index' con backend opcional."""
    if backend is None:
        cap = cv2.VideoCapture(index)
        tag = "default"
    else:
        cap = cv2.VideoCapture(index, backend)
        tag = f"backend={backend}"

    if not cap.isOpened():
        return None, f"❌ No abre: index={index}, {tag}"

    if FORCE_W:   cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FORCE_W)
    if FORCE_H:   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FORCE_H)
    if FORCE_FPS: cap.set(cv2.CAP_PROP_FPS,          FORCE_FPS)

    ok, _ = cap.read()
    if ok:
        return cap, f"✅ Cámara abierta: index={index}, {tag}"
    cap.release()
    return None, f"❌ isOpened=True pero read() falló: index={index}, {tag}"

def discover_cameras(max_cams=4):
    """Escanea índices y devuelve lista de (idx, cap or None) hasta max_cams."""
    found = []
    # orden de backends en Linux
    backends = []
    if _has('CAP_V4L2'):      backends.append(cv2.CAP_V4L2)
    if _has('CAP_GSTREAMER'): backends.append(cv2.CAP_GSTREAMER)
    backends.append(None)

    for idx in SCAN_RANGE:
        for be in backends:
            cap, msg = try_open(idx, be)
            print(msg)
            if cap is not None:
                found.append((idx, cap))
                break
        if len(found) >= max_cams:
            break

    # Rellena con None si hay menos de max_cams
    while len(found) < max_cams:
        found.append((None, None))
    return found[:max_cams]

def make_placeholder(text: str, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0,0), (w-1,h-1), (80,80,80), 2)
    cv2.putText(img, text, (20, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2, cv2.LINE_AA)
    return img

def label_tile(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return img

# ===== Descubrir hasta 4 cámaras =====
caps = discover_cameras(MAX_CAMERAS)  # [(idx, cap or None), ...] len=4

# ===== Ventana única =====
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TILE_W*2, TILE_H*2)
sleep(0.1)

def read_annotate_or_placeholder(idx, cap):
    """Lee frame, corre YOLO y devuelve tile (o placeholder si falla)."""
    if cap is None:
        return make_placeholder("No signal", TILE_W, TILE_H)

    ok, frame = cap.read()
    if not ok or frame is None:
        return make_placeholder(f"Cam {idx} sin señal", TILE_W, TILE_H)

    # Inference
    results = model(frame, size=INFER_SIZE)
    annotated = results.render()[0]
    tile = cv2.resize(annotated, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    label = f"Cam {idx}" if idx is not None else "Cam ?"
    return label_tile(tile, label)

# ===== Bucle principal =====
with torch.inference_mode():
    while True:
        tiles = [read_annotate_or_placeholder(idx, cap) for idx, cap in caps]

        top    = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], tiles[3]))
        mosaic = np.vstack((top, bottom))

        cv2.imshow(WIN_NAME, mosaic)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC / q
            break
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

# ===== Limpieza =====
for _, cap in caps:
    if cap is not None:
        cap.release()
cv2.destroyAllWindows()
