import http.server
import signal
import socket
import socketserver
import threading
from time import sleep

import cv2
import RPi.GPIO as GPIO

from guidance import GuidedDrive
from ball_chase import open_camera, chase_loop
from perception import build_debug_view

# Keep running if the SSH session drops - without this, losing the
# connection sends SIGHUP and the default reaction is to just exit.
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21

# No button wired up right now - set back to True once it is.
WAIT_FOR_BUTTON = False

# Serves a live view of what the chase loop sees, at http://<pi-ip>:8080/ -
# reuses the same frame the chase loop already reads (only one process can
# hold the webcam open at a time, so this can't run alongside
# cam_control.py). Set to False to skip it (e.g. once running for real).
SERVE_LIVE_VIEW = True
LIVE_VIEW_PORT = 8080
JPEG_QUALITY = 60

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
if WAIT_FOR_BUTTON:
    GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

_latest_jpeg = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)


def _on_frame(frame, masks):
    global _latest_jpeg, _frame_id
    ok, jpg = cv2.imencode(".jpg", build_debug_view(frame, masks), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
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
    drive = GuidedDrive()
    cap = None
    server = None
    try:
        if WAIT_FOR_BUTTON:
            print("Ready. Press the button to start...")
            while True:
                if GPIO.input(BTN_PIN) == GPIO.LOW:
                    print("Button pressed - starting...")
                    break
                sleep(0.05)

        cap = open_camera()
        if cap is None:
            return

        def _track_cap(new_cap):
            nonlocal cap
            cap = new_cap

        on_frame = None
        if SERVE_LIVE_VIEW:
            server = socketserver.ThreadingTCPServer(("0.0.0.0", LIVE_VIEW_PORT), _StreamingHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"Live view at http://<pi-ip-address>:{LIVE_VIEW_PORT}/")
            on_frame = _on_frame

        print("Chasing the ball. Ctrl+C to stop.")
        chase_loop(cap, drive, on_frame=on_frame, on_reconnect=_track_cap)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    finally:
        print("Cleaning up GPIO resources...")
        drive.stop()
        drive.cleanup()
        if cap is not None:
            cap.release()
        if server is not None:
            server.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
