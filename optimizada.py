# -*- coding: utf-8 -*-
"""
4 cámaras -> mosaico 2x2 con YOLOv5 custom (best.pt)
- Escanea SOLO /dev/video 0,2,4,6
- Fuerza MJPG 640x480 @15fps (ahorra ancho de banda USB)
- Baja carga: INFER_SIZE=416 + frame-skipping (1/3)
- Dibujo ligero con OpenCV (sin results.render())
- Una ventana; ESC/q para salir
"""

import os, warnings, platform, cv2, torch, numpy as np
from time import sleep
from pathlib import Path

# --- estabilidad de GUI Qt ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# --- reducir uso de CPU/hilos ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# ===== Config =====
WEIGHTS     = Path("./best.pt")   # tu modelo entrenado
SCAN_RANGE  = [0, 2, 4, 6]        # tus índices buenos
MAX_CAMERAS = 4
TILE_W, TILE_H = 640, 360
WIN_NAME    = f"best.pt (4 cams)"
INFER_SIZE  = 416
CONF_THRES  = 0.30
IOU_THRES   = 0.45
FORCE_W, FORCE_H = 640, 480
FORCE_FPS   = 15
FORCE_MJPG  = True
FRAME_SKIP  = 2  # infiere 1 de cada 3 frames por cámara

if not WEIGHTS.exists():
    raise FileNotFoundError(f"No se encontró {WEIGHTS.resolve()}")

print("OpenCV:", cv2.__version__, "| SO:", platform.system(), platform.release())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")
torch.set_num_threads(2)
try: cv2.setNumThreads(2)
except: pass

# ===== Cargar modelo YOLOv5 custom =====
model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path=str(WEIGHTS),
    source='github', trust_repo=True, force_reload=False
).to(DEVICE)
model.conf = CONF_THRES
model.iou  = IOU_THRES
model.max_det = 100

# ===== Utilidades =====
def _has(name: str) -> bool:
    return hasattr(cv2, name)

def _force_params(cap):
    if FORCE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if FORCE_W:   cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FORCE_W)
    if FORCE_H:   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FORCE_H)
    if FORCE_FPS: cap.set(cv2.CAP_PROP_FPS,          FORCE_FPS)

def try_open(index, backend=None):
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None, f"❌ No abre: /dev/video{index}"
    _force_params(cap)
    ok, _ = cap.read()
    if ok:
        w,h,fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        return cap, f"✅ /dev/video{index} OK {w}x{h}@{fps:.0f}"
    cap.release()
    return None, f"❌ read() falló: /dev/video{index}"

def discover_cameras(max_cams=4):
    found = []
    backends = []
    if _has('CAP_V4L2'): backends.append(cv2.CAP_V4L2)
    if _has('CAP_GSTREAMER'): backends.append(cv2.CAP_GSTREAMER)
    backends.append(None)

    for idx in SCAN_RANGE:
        for be in backends:
            cap, msg = try_open(idx, be)
            print(msg)
            if cap is not None:
                found.append((idx, cap))
                break
        if len(found) >= max_cams: break

    while len(found) < max_cams: found.append((None,None))
    return found[:max_cams]

def make_placeholder(text,w,h):
    img = np.zeros((h,w,3),dtype=np.uint8)
    cv2.rectangle(img,(0,0),(w-1,h-1),(80,80,80),2)
    cv2.putText(img,text,(20,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,200,200),2,cv2.LINE_AA)
    return img

def label_tile(img,text):
    cv2.putText(img,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(img,text,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
    return img

def draw_dets(frame, results):
    d = results.xyxy[0].detach().cpu().numpy()
    for (x1,y1,x2,y2,conf,cls) in d:
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,label,(x1,max(0,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,label,(x1,max(0,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    return frame

# ===== Cámaras =====
caps = discover_cameras(MAX_CAMERAS)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, TILE_W*2, TILE_H*2)
sleep(0.1)

last_annotated, frame_counters = {}, {}
def process_tile(idx,cap):
    if cap is None: return make_placeholder("No signal",TILE_W,TILE_H)
    c = frame_counters.get(idx,0); do_infer = (c % (FRAME_SKIP+1) == 0)
    frame_counters[idx] = c+1
    ok,frame = cap.read()
    if not ok or frame is None: return make_placeholder(f"Cam {idx} sin señal",TILE_W,TILE_H)
    if do_infer:
        with torch.inference_mode():
            results = model(frame, size=INFER_SIZE)
        annotated = draw_dets(frame, results)
        last_annotated[idx] = annotated
    else:
        annotated = last_annotated.get(idx, frame)
    tile = cv2.resize(annotated,(TILE_W,TILE_H))
    return label_tile(tile, f"Cam {idx}")

# ===== Loop =====
with torch.inference_mode():
    while True:
        tiles = [process_tile(idx,cap) for idx,cap in caps]
        mosaic = np.vstack((np.hstack((tiles[0],tiles[1])),
                            np.hstack((tiles[2],tiles[3]))))
        cv2.imshow(WIN_NAME,mosaic)
        k = cv2.waitKey(1)&0xFF
        if k in (27, ord('q')): break
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break

for _,cap in caps:
    if cap is not None: cap.release()
cv2.destroyAllWindows()
