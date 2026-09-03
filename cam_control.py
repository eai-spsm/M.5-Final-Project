import http.server
import socketserver
import threading
import time

import cv2

# Kept low since this is streamed over an SSH tunnel (VS Code Remote-SSH /
# Simple Browser) - a smaller, lower-quality, lower-FPS stream is much less
# laggy than a big high-quality one over that kind of link.
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 60      # 0-100, lower = smaller/faster, blockier
TARGET_FPS = 12
PORT = 8080

# Latest frame + a version counter, shared between the capture thread and
# any number of viewers. Viewers wait on _new_frame instead of polling, so
# they only ever send an actually-new frame - no duplicate resends.
_latest_jpeg = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)


def capture_loop(cap):
    global _latest_jpeg, _frame_id
    frame_interval = 1.0 / TARGET_FPS
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        with _new_frame:
            _latest_jpeg = jpg.tobytes()
            _frame_id += 1
            _new_frame.notify_all()

        elapsed = time.time() - start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (
                b"<html><body style='margin:0;background:#111'>"
                b"<img src='/stream' style='width:100%;display:block' />"
                b"</body></html>"
            )
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
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab/connection
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence the default per-request access log


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    threading.Thread(target=capture_loop, args=(cap,), daemon=True).start()

    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), StreamingHandler) as server:
        print(f"Live view at http://<pi-ip-address>:{PORT}/  (Ctrl+C to stop)")
        print("In VS Code: Ctrl+Shift+P -> 'Browser: Open Integrated Browser' -> paste that URL.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()


if __name__ == "__main__":
    main()
