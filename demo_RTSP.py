import cv2
import torch
import threading
from queue import Queue

from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

# -------- CONFIG --------
RTSP_URL = "rtsp://admin:1234567a@192.168.100.107:554/cam/realmonitor?channel=1&subtype=0"
# https://github.com/YapaLab/yolo-face?tab=readme-ov-file
# DETECTOR_WEIGHTS = "models/yolov12n-face.pt"
# DETECTOR_WEIGHTS = "models/yolov8x_person_face.pt"
DETECTOR_WEIGHTS = "models/scrfd.onnx"

# DETECTOR_WEIGHTS = "models/yolov8n.pt"
CHECKPOINT = "models/mivolo_imbd.pth.tar"
DEVICE = "cpu"  # "cuda" if available
# ------------------------


class Args:
    detector_weights = DETECTOR_WEIGHTS
    checkpoint = CHECKPOINT
    with_persons = True
    disable_faces = False
    draw = True
    device = DEVICE
    input = None
    output = None


def capture_loop(cap, frame_queue, stop_event):
    """Continuously capture frames (latest only)"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get_nowait()  # drop old frame

        frame_queue.put(frame)


def main():
    setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    predictor = Predictor(Args, verbose=True)

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
        frame = frame_queue.get()  # always latest frame

        _, output_frame = predictor.recognize(frame)
        output_frame_resize = cv2.resize(output_frame, (0, 0), fx=0.5, fy=0.5)

        # cv2.imshow("MiVOLO RTSP (Threaded)", output_frame)
        cv2.imshow("MiVOLO RTSP (Resized)", output_frame_resize)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    stop_event.set()
    capture_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
