from unittest import result
from scrfd import SCRFD, Threshold
from PIL import Image
import cv2
import threading
from queue import Queue
import time 
RTSP_URL = "rtsp://admin:1234567a@192.168.100.107:554/cam/realmonitor?channel=1&subtype=0"

face_detector = SCRFD.from_path("models/scrfd.onnx")
threshold = Threshold(probability=0.4)

def capture_loop(cap, frame_queue, stop_event):
    """Continuously capture frames (latest only)"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get_nowait()  # drop old frame

        frame_queue.put(frame)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open RTSP stream")
frame_queue = Queue(maxsize=1)
stop_event = threading.Event()

capture_thread = threading.Thread(
    target=capture_loop,
    args=(cap, frame_queue, stop_event),
    daemon=True
)
capture_thread.start()

print("✅ RTSP stream started")

while True:
    frame = frame_queue.get()  # NumPy array (BGR)

    # Convert OpenCV frame -> PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Face detection
    t0 = time.time()
    faces  = face_detector.detect(frame_pil, threshold=threshold)
    print(f"Detection time: {(time.time() - t0)*1000:.0f} msec")

    # Draw results back on OpenCV frame
    for face in faces:
        bbox = face.bbox
        print(f"{bbox}")
        # Extract coordinates
        upper_left = (int(bbox.upper_left.x), int(bbox.upper_left.y))
        lower_right = (int(bbox.lower_right.x), int(bbox.lower_right.y))

        # Draw rectangle on frame
        cv2.rectangle(frame, upper_left, lower_right, (0, 255, 0), 2)
        output_frame_resize = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Face Detection", output_frame_resize)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


stop_event.set()
capture_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
    
    
