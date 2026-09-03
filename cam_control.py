import http.server
import socketserver
import threading

import cv2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PORT = 8080

# Latest frame, shared between the capture thread and any number of viewers
_latest_jpeg = None
_lock = threading.Lock()


def capture_loop(cap):
    global _latest_jpeg
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        with _lock:
            _latest_jpeg = jpg.tobytes()


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
            try:
                while True:
                    with _lock:
                        jpg = _latest_jpeg
                    if jpg is None:
                        continue
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
        print("In VS Code: Ctrl+Shift+P -> 'Simple Browser: Show' -> paste that URL.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()


if __name__ == "__main__":
    main()
