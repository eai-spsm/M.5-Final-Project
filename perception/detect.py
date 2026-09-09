import http.server
import socket
import socketserver
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "data" / "best2.pt"
CONF_THRESHOLD = 0.5
IMG_SIZE = 320       # smaller inference size = faster on Pi CPU
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
SKIP_FACTOR = 2       # only run inference on every (SKIP_FACTOR + 1)th frame
BOX_COLOR = (0, 255, 0)
JPEG_QUALITY = 60
PORT = 8082

# No HDMI on this Pi, so cv2.imshow() wouldn't show anything - same MJPEG
# pattern as cam_control.py instead. Pure detection viewer: no GPIO/motor
# code, doesn't touch movement or guidance at all.

_latest_jpeg = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)


def _publish(frame):
    global _latest_jpeg, _frame_id
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return
    with _new_frame:
        _latest_jpeg = jpg.tobytes()
        _frame_id += 1
        _new_frame.notify_all()


class _StreamingHandler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%;display:block' /></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            last_sent_id = None
            try:
                while True:
                    with _new_frame:
                        while _frame_id == last_sent_id or _latest_jpeg is None:
                            _new_frame.wait()
                        jpg = _latest_jpeg
                        last_sent_id = _frame_id
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    server = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), _StreamingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Detection view at http://<pi-ip-address>:{PORT}/  (Ctrl+C to stop)")

    frame_count = 0
    last_boxes = []
    last_detected = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if frame_count % (SKIP_FACTOR + 1) == 0:
                try:
                    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
                    boxes = results[0].boxes

                    detected = set()
                    last_boxes = []
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        name = model.names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        last_boxes.append((x1, y1, x2, y2, f"{name} {conf:.2f}"))
                        detected.add(name)
                        if name not in last_detected:
                            print(f"Detected {name} ({conf:.2f})")
                    last_detected = detected
                except Exception as frame_err:
                    print(f"Error processing frame: {frame_err}")

            for x1, y1, x2, y2, label in last_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)
                cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1)

            frame_count += 1
            _publish(frame)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        server.shutdown()


if __name__ == "__main__":
    main()
