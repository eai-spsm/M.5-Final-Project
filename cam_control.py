import http.server
import socket
import socketserver
import threading
import time

import cv2

from perception import build_debug_view

# Kept low since this is streamed over an SSH tunnel (VS Code Remote-SSH /
# Simple Browser) - a smaller, lower-quality, lower-FPS stream is much less
# laggy than a big high-quality one over that kind of link.
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 60      # 0-100, lower = smaller/faster, blockier
TARGET_FPS = 12
PORT = 8080

# Latest frame (color + grayscale + segmentation debug) + a version
# counter, shared between the cappture thread and any number of viewers.
# One cv2.VideoCapture read per cycle feeds all three views, since most
# webcams only allow one process to hold the device open at a time -
# running cam_control.py and a separate segmentation script at once would
# fight over the camera instead of sharing it.
_latest_color = None
_latest_gray = None
_latest_segment = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)


def capture_loop(cap):
    global _latest_color, _latest_gray, _latest_segment, _frame_id
    frame_interval = 1.0 / TARGET_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        ok_c, jpg_color = cv2.imencode(".jpg", frame, encode_params)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_g, jpg_gray = cv2.imencode(".jpg", gray, encode_params)
        ok_s, jpg_segment = cv2.imencode(".jpg", build_debug_view(frame), encode_params)
        if not (ok_c and ok_g and ok_s):
            continue

        with _new_frame:
            _latest_color = jpg_color.tobytes()
            _latest_gray = jpg_gray.tobytes()
            _latest_segment = jpg_segment.tobytes()
            _frame_id += 1
            _new_frame.notify_all()

        elapsed = time.time() - start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


def _stream(wfile, get_jpeg):
    last_sent_id = None
    while True:
        with _new_frame:
            while _frame_id == last_sent_id or get_jpeg() is None:
                _new_frame.wait()
            jpg = get_jpeg()
            last_sent_id = _frame_id

        wfile.write(b"--FRAME\r\n")
        wfile.write(b"Content-Type: image/jpeg\r\n")
        wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
        wfile.write(jpg)
        wfile.write(b"\r\n")


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        # Disable Nagle's algorithm - without this, TCP can hold small
        # writes (like each JPEG chunk here) for tens of ms trying to
        # bundle them, which adds up to real, visible stream lag.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (
                b"<html><body style='margin:0;background:#111;display:flex;flex-wrap:wrap'>"
                b"<div><p style='color:#aaa;font:12px sans-serif;margin:4px'>Color</p>"
                b"<img src='/stream' style='width:100%;display:block' /></div>"
                b"<div><p style='color:#aaa;font:12px sans-serif;margin:4px'>Grayscale</p>"
                b"<img src='/stream_gray' style='width:100%;display:block' /></div>"
                b"<div style='flex-basis:100%'>"
                b"<p style='color:#aaa;font:12px sans-serif;margin:4px'>"
                b"Segmentation debug - top-left: ball detection | top-right: ball cut-out | "
                b"bottom-left: wall cut-out | bottom-right: floor cut-out</p>"
                b"<img src='/stream_segment' style='width:100%;display:block' /></div>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/stream", "/stream_gray", "/stream_segment"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            getters = {
                "/stream": lambda: _latest_color,
                "/stream_gray": lambda: _latest_gray,
                "/stream_segment": lambda: _latest_segment,
            }
            try:
                _stream(self.wfile, getters[self.path])
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
    # MJPG capture format + a 1-frame buffer: many USB webcams default to a
    # slow raw format and buffer several frames internally unless told
    # otherwise, both of which add startup/streaming lag.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    threading.Thread(target=capture_loop, args=(cap,), daemon=True).start()

    socketserver.ThreadingTCPServer.allow_reuse_address = True  # avoid "Address already in use" on a quick restart
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
