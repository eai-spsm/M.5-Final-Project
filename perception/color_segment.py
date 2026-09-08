import http.server
import socket
import socketserver
import threading
import time

import cv2
import numpy as np

# PLACEHOLDER HSV ranges - tune these on-site under the real match lighting.
# OpenCV hue is 0-179 (not 0-359). To find good ranges: run this script,
# open /raw in the browser, and adjust these numbers while watching /masks
# until each target is cleanly picked out with minimal noise.
BALL_LOW = np.array([35, 80, 60])     # bright green ball
BALL_HIGH = np.array([85, 255, 255])

WALL_LOW = np.array([0, 0, 0])        # black wall / forbidden-zone tape
WALL_HIGH = np.array([179, 255, 60])

FLOOR_LOW = np.array([0, 0, 90])      # gray/white tile floor (low saturation)
FLOOR_HIGH = np.array([179, 60, 255])

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 60
TARGET_FPS = 12
PORT = 8081


def get_masks(frame):
    """HSV-threshold + bitwise ops to separate ball / wall / floor / the
    rest. Returns a dict of single-channel masks (255 = belongs to that
    class, 0 = doesn't), all mutually exclusive.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ball = cv2.inRange(hsv, BALL_LOW, BALL_HIGH)
    wall = cv2.inRange(hsv, WALL_LOW, WALL_HIGH)
    floor = cv2.inRange(hsv, FLOOR_LOW, FLOOR_HIGH)

    # Ranges can overlap slightly at their edges - bitwise_and each mask
    # against the inverse of ball's (ball wins ties) so every pixel ends up
    # in at most one class.
    not_ball = cv2.bitwise_not(ball)
    wall = cv2.bitwise_and(wall, not_ball)
    floor = cv2.bitwise_and(floor, not_ball)
    floor = cv2.bitwise_and(floor, cv2.bitwise_not(wall))

    classified = cv2.bitwise_or(cv2.bitwise_or(ball, wall), floor)
    unclassified = cv2.bitwise_not(classified)

    return {"ball": ball, "wall": wall, "floor": floor, "unclassified": unclassified}


def cut_out(frame, mask):
    # bitwise_and against a mask: keep the real pixels where mask==255,
    # black out everywhere else. This is the "cut out an area" operation.
    return cv2.bitwise_and(frame, frame, mask=mask)


def ball_center(mask):
    # Largest ball-colored contour's centroid, or None if nothing found.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:  # ignore tiny specks/noise
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def build_debug_view(frame):
    masks = get_masks(frame)

    # Colors picked to be visually distinct from the real colors, purely
    # for this debug view - ball True color already shows in "raw".
    wall_view = cut_out(frame, masks["wall"])
    floor_view = cut_out(frame, masks["floor"])
    ball_view = cut_out(frame, masks["ball"])

    center = ball_center(masks["ball"])
    annotated = frame.copy()
    if center is not None:
        cv2.circle(annotated, center, 8, (0, 0, 255), 2)
        cv2.putText(annotated, "ball", (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    top = np.hstack([annotated, ball_view])
    bottom = np.hstack([wall_view, floor_view])
    return np.vstack([top, bottom])


# --- MJPEG viewer, same pattern as cam_control.py ---------------------

_latest_jpeg = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)


def capture_loop(cap):
    global _latest_jpeg, _frame_id
    frame_interval = 1.0 / TARGET_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        debug_frame = build_debug_view(frame)
        ok, jpg = cv2.imencode(".jpg", debug_frame, encode_params)
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
    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (
                b"<html><body style='margin:0;background:#111'>"
                b"<p style='color:#aaa;font:12px sans-serif;margin:4px'>"
                b"Top-left: ball detection | Top-right: ball cut-out | "
                b"Bottom-left: wall cut-out | Bottom-right: floor cut-out</p>"
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    threading.Thread(target=capture_loop, args=(cap,), daemon=True).start()

    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), StreamingHandler) as server:
        print(f"Color segmentation debug view at http://<pi-ip-address>:{PORT}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()


if __name__ == "__main__":
    main()
