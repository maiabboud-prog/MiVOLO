
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase ,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import cv2
import threading
from queue import Queue
import time 
import numpy as np

# RTSP_URL = "rtsp://admin:1234567a@192.168.100.107:554/cam/realmonitor?channel=1&subtype=0"
VIDEO_PATH = "C:\\Users\\Manar\\Documents\\Library\\videos_library\\10AM\\NVR-1993_ch5_main_20260113100004_20260113110004.mp4"

def capture_loop(cap, frame_queue, stop_event):
    """Continuously capture frames (latest only)"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get_nowait()  # drop old frame

        frame_queue.put(frame)

cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open RTSP stream")
frame_queue = Queue(maxsize=1)
stop_event = threading.Event()

capture_thread = threading.Thread(
    target=capture_loop,
    args=(cap, frame_queue, stop_event),
    daemon=True
)

start_frame = 20*25*60  # zero-based
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
capture_thread.start()

print("✅ RTSP stream started")
# model = RFDETRBase()
model = RFDETRNano()
# model.optimize_for_inference()

while True:
    frame = frame_queue.get()  # NumPy array (BGR)
    frame_pil = Image.fromarray(frame)

    t0 = time.time()
    detections = model.predict(frame_pil, threshold=0.2)
    print(f"Inference time: {(time.time() - t0)*1000:.0f} milliseconds")
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
]

    annotated_image = frame_pil.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    annotated_bgr = cv2.cvtColor(np.array(annotated_image),cv2.COLOR_RGB2RGBA)
    output_frame_resize = cv2.resize(annotated_bgr, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("RFDETR Video Stream", output_frame_resize)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_event.set()
capture_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
