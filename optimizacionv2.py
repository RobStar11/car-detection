# -*- coding: utf-8 -*-
"""
Visor 1-cámara conmutada (ahorra CPU/USB) + YOLOv5 (best.pt)
- Teclas: 1,2,3,4 -> activa /dev/video[0,2,4,6]
- 'y' activa/desactiva detección en vivo
- 'q' o ESC para salir
- Abre SOLO la cámara activa; las demás están cerradas => mínimo consumo
"""

import os, warnings, platform, cv2, torch, numpy as np
from pathlib import Path
from time import time

# --- GUI estable en X11 (Ubuntu) ---
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

# --- Reducir hilos para bajar CPU ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# ===== Config =====
WEIGHTS      = Path("./best.pt")       # tu modelo YOLOv5 entrenado
CAM_MAP      = {1:0, 2:2, 3:4, 4:6}    # teclas -> índices V4L2
DEFAULT_KEY  = 1                       # arranca en /dev/video0
FORCE_WH     = (640, 480)              # 640x480 para fluidez
FORCE_FPS    = 15                      # 15 fps (sube a 30 si aguanta)
FORCE_MJPG   = True                    # MJPG = menos CPU para decodificar
INFER_SIZE   = 320                     # entrada YOLO (menos = más fluido)
CONF_THRES   = 0.30
IOU_THRES    = 0.45
MAX_DET      = 75
FRAME_SKIP   = 2                       # infiere 1 de cada (FRAME_SKIP+1)
WINDOW_NAME  = "One-Cam Viewer + YOLO (switch 1-4, Y on/off)"

if not WEIGHTS.exists():
    raise FileNotFoundError(f"No se encontró {WEIGHTS.resolve()}")

print("OpenCV:", cv2.__version__, "| SO:", platform.system(), platform.release())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ===== Cargar YOLOv5 =====
model = torch.hub.load(
    'ultralytics/yolov5', 'custom',
    path=str(WEIGHTS), source='github', trust_repo=True, force_reload=False
).to(DEVICE)
model.conf = CONF_THRES
model.iou  = IOU_THRES
model.max_det = MAX_DET

if DEVICE == "cuda":
    model.half()
    TORCH_DTYPE = np.float16
else:
    TORCH_DTYPE = np.float32
torch.set_num_threads(1)
try: cv2.setNumThreads(1)
except: pass

USE_V4L2 = hasattr(cv2, 'CAP_V4L2')

def open_cam(index: int):
    """Abre una cámara con parámetros livianos."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if USE_V4L2 else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    if FORCE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    w,h = FORCE_WH
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          FORCE_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # baja latencia
    except Exception:
        pass
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    ww,hh,fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    print(f"✅ Abierta /dev/video{index}: {ww}x{hh}@{fps:.0f}")
    return cap

def close_cam(cap):
    if cap is not None:
        try: cap.release()
        except: pass

def draw_help(frame, cam_idx, infer_on, fps_txt):
    h, w = frame.shape[:2]
    txt1 = f"Cam /dev/video{cam_idx} | {'YOLO:ON' if infer_on else 'YOLO:OFF'} | {fps_txt}"
    txt2 = "Switch: [1]=0, [2]=2, [3]=4, [4]=6 | Toggle YOLO: [Y] | Quit: [Q]/[ESC]"
    cv2.putText(frame, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, txt2, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt2, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return frame

def infer_and_draw(frame):
    """Inferimos en copia downscaled (INFER_SIZE) y dibujamos cajas escaladas."""
    H, W = frame.shape[:2]
    img = cv2.resize(frame, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]                          # BGR->RGB
    img = img.astype(TORCH_DTYPE) / 255.0
    img = np.transpose(img, (2,0,1))[None, ...]    # NCHW
    t = torch.from_numpy(img).to(DEVICE, non_blocking=True)

    if DEVICE == "cuda":
        with torch.cuda.amp.autocast(True), torch.inference_mode():
            out = model(t, size=INFER_SIZE)
    else:
        with torch.inference_mode():
            out = model(t, size=INFER_SIZE)

    det = out.xyxy[0].detach().to('cpu').numpy() if hasattr(out, "xyxy") else np.empty((0,6))
    sx, sy = W/INFER_SIZE, H/INFER_SIZE
    for (x1,y1,x2,y2,conf,cls) in det:
        x1,y1,x2,y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        name = model.names.get(int(cls), str(int(cls))) if hasattr(model, "names") else str(int(cls))
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame

# ===== Main loop (una cámara activa a la vez) =====
active_key  = DEFAULT_KEY
active_idx  = CAM_MAP[active_key]
cap         = open_cam(active_idx)
infer_on    = True
frame_i     = 0
fps_avg_t   = time()
fps_disp    = "FPS: --"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, FORCE_WH[0], FORCE_WH[1])

while True:
    if cap is None:
        # Si no abrió, muestra placeholder y espera tecla de cambio
        frame = np.zeros((FORCE_WH[1], FORCE_WH[0], 3), dtype=np.uint8)
        cv2.putText(frame, f"No se pudo abrir /dev/video{active_idx}", (20, FORCE_WH[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        frame = draw_help(frame, active_idx, infer_on, fps_disp)
        cv2.imshow(WINDOW_NAME, frame)
        k = cv2.waitKey(10) & 0xFF
        if k in (27, ord('q')):
            break
        if k in (ord('1'), ord('2'), ord('3'), ord('4')):
            new_key = int(chr(k))
            if new_key in CAM_MAP and new_key != active_key:
                close_cam(cap)
                active_key = new_key
                active_idx = CAM_MAP[active_key]
                cap = open_cam(active_idx)
        if k in (ord('y'), ord('Y')):
            infer_on = not infer_on
        continue

    # Lectura con baja latencia
    if not cap.grab():
        continue
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        continue

    # Detección ligera cada N frames
    do_infer = infer_on and (frame_i % (FRAME_SKIP + 1) == 0)
    frame_i += 1

    if do_infer:
        frame = infer_and_draw(frame)

    # FPS simple (visual)
    if frame_i % 15 == 0:
        now = time()
        fps = 15.0 / max(1e-3, (now - fps_avg_t))
        fps_avg_t = now
        fps_disp = f"FPS: {fps:.1f}"

    frame = draw_help(frame, active_idx, infer_on, fps_disp)
    cv2.imshow(WINDOW_NAME, frame)

    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

    # Cambiar cámara con 1/2/3/4
    if k in (ord('1'), ord('2'), ord('3'), ord('4')):
        new_key = int(chr(k))
        if new_key in CAM_MAP and new_key != active_key:
            close_cam(cap)                    # CERRAR la actual
            active_key = new_key
            active_idx = CAM_MAP[active_key]
            cap = open_cam(active_idx)        # ABRIR la nueva
            frame_i = 0                       # reset contadores/FPS

    # Toggle YOLO
    if k in (ord('y'), ord('Y')):
        infer_on = not infer_on

close_cam(cap)
cv2.destroyAllWindows()
