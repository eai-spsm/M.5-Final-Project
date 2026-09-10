import http.server
import signal
import socket
import socketserver
import threading
import time

import cv2
import RPi.GPIO as GPIO

from guidance import GuidedDrive
from perception import get_masks, find_goal_gap, ball_angle_offset, ball_center, build_debug_view

# Single entry point: camera + perception + drive state machine + live
# view, all in one file. Perception is deliberately just HSV color
# (perception/color_segment.py: get_masks/ball_center/ball_angle_offset/
# find_goal_gap) - no YOLO, no ArUco, no separate detector modules. That
# color-only approach is what's actually been working well; everything
# else tried on top of it added latency/complexity without a clear win.

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 3-tier priority, checked in this order every step:
#   1. Find/chase the ball (default behavior)
#   2. Evade an incoming threat from behind (overrides #1 when triggered)
#   3. Don't go into the goal (overrides both #1 and #2 - checked first,
#      since driving into the goal is the one thing that must never happen
#      regardless of what else is going on)
#
# No position/heading-based backup checks (Navigator's dead reckoning in
# guidance/guided_drive.py isn't calibrated yet - it was firing "near
# goal"/"near wall" early on bad position estimates alone), no
# vision-based opponent-bot detection (that needed YOLO), no pin-stuck/
# unpin recovery. Vision (HSV + the wall mask) and the rear ultrasonic are
# the only signals.

# Don't bother turning for offsets smaller than this - avoids twitching
# back and forth over tiny, noisy angle estimates when the ball's roughly
# ahead already.
CENTERED_TOLERANCE_DEG = 8

# Duty cycle % used while spinning to search for the ball - much slower
# than a normal turn (default speed is 60) so a frame doesn't blur past
# the ball/wall/goal and miss it.
SEARCH_SPEED = 30

# If the ball was found last frame but isn't this frame (HSV flicker,
# motion blur, a one-frame occlusion), keep pursuing its last-known
# bearing for up to this many consecutive missed frames instead of
# immediately giving up and spinning into a full search - see the
# "held" branch in chase_step.
LOST_BALL_GRACE_FRAMES = 5

# The wall is low enough that background well outside the field (chairs,
# bags, anything else green) is visible above it in frame - that's most
# of the "noise" HSV was picking up. Exclude everything above this row
# from ball detection entirely (see _ball_roi). Tradeoff: a real ball far
# enough away also sits higher in frame (perspective), so this will blind
# the detector to a distant ball too - the search sweep +
# LOST_BALL_GRACE_FRAMES hold still cover that once it gets closer.
# PLACEHOLDER - tune against where the wall top/background boundary
# actually sits in frame.
BALL_ROI_Y_START = int(FRAME_HEIGHT * 0.35)

# If the wall mask covers this much of the frame, we're facing straight
# into it (or nearly touching it) - back up instead of trying to chase
# whatever the ball logic thinks it sees. Cheap check (reuses the wall
# mask already computed for the goal-gap check below).
WALL_COVERAGE_THRESHOLD = 0.80

# The ultrasonic sensor is mounted at the REAR - it watches for something
# closing in from behind (e.g. an opponent bot approaching while we're
# facing/pushing the ball), not a front obstruction. It can't say which
# side a rear threat is on, and strafing wouldn't increase distance from
# something behind us anyway - push forward instead, which does.
REAR_THREAT_DISTANCE_CM = 15
EVADE_SPEED = 45

# How wide the detected goal gap needs to be (as a fraction of frame width)
# before treating it as "close enough to stop" - matches how big the
# opening looks right before the goal mouth.
GOAL_GAP_CLOSE_WIDTH_FRACTION = 0.5

# After spinning this many degrees without finding the ball, assume it's
# not visible from here and nudge forward before continuing the search,
# instead of spinning in the same spot forever. Slightly under 360 to
# account for tracking imprecision.
SEARCH_FULL_SWEEP_DEG = 350

# If the camera read fails this many times in a row, assume it's actually
# disconnected (not just a one-off dropped frame) and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21
WAIT_FOR_BUTTON = False  # no button wired up right now - set True once it is

# Serves a live view of what the chase loop sees, at http://<pi-ip>:8080/ -
# reuses the same frame the chase loop already reads (only one process can
# hold the webcam open at a time, so this can't run alongside
# cam_control.py). Set to False to skip it.
SERVE_LIVE_VIEW = True
LIVE_VIEW_PORT = 8080
JPEG_QUALITY = 60


# movement/movement.py's set_wheels() flips the H-bridge direction pins
# directly with no braking step - reversing straight from one direction to
# its exact opposite (forward<->backward) at full duty cycle spikes
# current/back-EMF across all 4 motors at once hard enough to brown out
# the Pi's power rail, which can reset the Wi-Fi chip (or the whole Pi)
# and looks exactly like a dropped SSH session. _safe_move forces one
# coast-to-a-stop step before any such reversal instead of flipping
# directly - costs at most one extra step's worth of delay. (rotate_to()
# is not covered here - it already ends with its own stop(), but a
# reversal *inside* a single rotate_to call, or between a raw
# rotate_right/left and a following rotate_to, isn't guarded; lower risk
# in practice since search sweeps are slow and turns are usually small.)
_OPPOSITES = {"forward": "backward", "backward": "forward"}


def _stop(drive, search_state):
    drive.stop()
    search_state["last_move"] = "stop"


def _safe_move(drive, search_state, move, speed=None):
    # Returns True if the move actually executed, False if it braked
    # instead (caller should report that and just retry next step).
    if _OPPOSITES.get(search_state.get("last_move")) == move:
        _stop(drive, search_state)
        return False
    getattr(drive, move)(speed=speed)
    search_state["last_move"] = move
    return True


def _ball_roi(mask):
    # Zeroes out rows above BALL_ROI_Y_START (a copy - never mutates the
    # shared mask other checks use) so ball_center() only ever sees the
    # floor region. Zeroing instead of cropping keeps returned coordinates
    # valid against the original frame, no offset math needed.
    roi = mask.copy()
    roi[:BALL_ROI_Y_START, :] = 0
    return roi


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def chase_step(cap, drive, on_frame=None, search_state=None):
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    if on_frame is not None:
        on_frame(frame, masks)

    wall_fraction = float((masks["wall"] > 0).mean())
    if wall_fraction >= WALL_COVERAGE_THRESHOLD:
        search_state["searching"] = False
        if _safe_move(drive, search_state, "backward"):
            return f"Wall fills {wall_fraction * 100:.0f}% of view - backing up"
        return f"Wall fills {wall_fraction * 100:.0f}% of view - braking before backing up"

    # --- Priority 3: don't go into the goal --------------------------
    gap = find_goal_gap(masks["wall"])
    close_by_width = gap is not None and gap["width_px"] >= GOAL_GAP_CLOSE_WIDTH_FRACTION * frame.shape[1]
    if close_by_width:
        search_state["searching"] = False
        if _safe_move(drive, search_state, "backward"):
            return "Near goal - backing out"
        return "Near goal - braking before backing out"

    center = ball_center(_ball_roi(masks["ball"]))
    if gap is not None and center is not None:
        gap_left = gap["center_x"] - gap["width_px"] / 2
        gap_right = gap["center_x"] + gap["width_px"] / 2
        if gap_left <= center[0] <= gap_right:
            # The ball itself is sitting inside the goal opening - don't
            # chase it in.
            search_state["searching"] = False
            if _safe_move(drive, search_state, "backward"):
                return "Ball is in the goal opening - backing out"
            return "Ball is in the goal opening - braking before backing out"

    # --- Priority 2: evade an incoming rear threat ---------------------
    distance = drive.get_distance()
    rear_threat = distance is not None and distance < REAR_THREAT_DISTANCE_CM
    if rear_threat:
        # Rear-facing ultrasonic - something's closing in from behind. It
        # can't say which side, and strafing wouldn't increase distance
        # from a rear threat anyway - push forward instead, which does.
        search_state["searching"] = False
        if _safe_move(drive, search_state, "forward", EVADE_SPEED):
            return f"Rear threat at {distance:.0f}cm - pushing forward"
        return f"Rear threat at {distance:.0f}cm - braking before pushing forward"

    # --- Priority 1: find/chase the ball -------------------------------
    held = False
    if center is not None:
        # Found it - lock in the world-frame bearing so a brief miss next
        # frame (HSV flicker, motion blur, a one-frame occlusion) can keep
        # heading the same direction instead of instantly giving up and
        # spinning into a full search.
        search_state["searching"] = False
        search_state["ball_miss_streak"] = 0
        angle = ball_angle_offset(center[0], frame.shape[1])
        search_state["last_ball_target_heading"] = (drive.pose()[2] + angle) % 360
    else:
        miss_streak = search_state.get("ball_miss_streak", 0) + 1
        search_state["ball_miss_streak"] = miss_streak
        target_heading = search_state.get("last_ball_target_heading")
        if target_heading is not None and miss_streak <= LOST_BALL_GRACE_FRAMES:
            # Still within the grace window - keep pursuing the last-known
            # bearing rather than treating this as a real loss yet.
            angle = (target_heading - drive.pose()[2] + 180) % 360 - 180
            held = True
        else:
            heading = drive.pose()[2]
            if not search_state.get("searching"):
                search_state["searching"] = True
                search_state["swept_deg"] = 0.0
            else:
                delta = (heading - search_state["last_heading"] + 180) % 360 - 180
                search_state["swept_deg"] += abs(delta)
            search_state["last_heading"] = heading

            drive.rotate_right(speed=SEARCH_SPEED)
            search_state["last_move"] = "rotate"  # not forward/backward - clears stale linear-move tracking
            swept = search_state["swept_deg"]

            if swept >= SEARCH_FULL_SWEEP_DEG:
                _safe_move(drive, search_state, "forward", SEARCH_SPEED)
                search_state["searching"] = False
                return "Full 360 sweep, no ball found - repositioning..."

            return f"Searching (spinning, {swept:.0f}/360 deg)..."

    held_note = " (held)" if held else ""
    if abs(angle) > CENTERED_TOLERANCE_DEG:
        target = (drive.pose()[2] + angle) % 360
        drive.rotate_to(target)
        search_state["last_move"] = "rotate"
        return f"Ball at {angle:+5.1f} deg - turning{held_note}"

    if _safe_move(drive, search_state, "forward"):
        return f"Ball centered ({angle:+5.1f} deg) - approaching{held_note}"
    return f"Ball centered ({angle:+5.1f} deg) - braking before approaching{held_note}"


def _safe_print(*args, **kwargs):
    # If stdout is gone (e.g. an SSH session dropped), printing raises
    # BrokenPipeError/OSError - that's fine to ignore, the driving logic
    # doesn't need anyone watching to keep working.
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def chase_loop(cap, drive, on_frame=None, on_reconnect=None):
    consecutive_failures = 0
    search_state = {}
    while True:
        start = time.time()
        status = chase_step(cap, drive, on_frame=on_frame, search_state=search_state)
        elapsed_ms = (time.time() - start) * 1000

        if status == "Camera read failed":
            consecutive_failures += 1
            if consecutive_failures >= CAMERA_RECONNECT_AFTER:
                _safe_print(f"\r{'Camera lost - reconnecting...':<60}", end="", flush=True)
                cap.release()
                time.sleep(CAMERA_RECONNECT_RETRY_DELAY)
                reopened = open_camera()
                if reopened is not None:
                    cap = reopened
                    consecutive_failures = 0
                    if on_reconnect is not None:
                        on_reconnect(cap)
                continue
        else:
            consecutive_failures = 0

        # Timing appended here (rather than baked into chase_step's return
        # value) so it can be correlated against real robot movement: if
        # this is, say, 150ms/step and the robot covers real ground in
        # that time, every command is being issued against an
        # already-stale picture of the world by the time it's acted on.
        _safe_print(f"\r{status} [{elapsed_ms:.0f}ms/step]{'':<20}", end="", flush=True)


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


_latest_jpeg = None
_frame_id = 0
_lock = threading.Lock()
_new_frame = threading.Condition(_lock)

# Keep running if the SSH session drops - without this, losing the
# connection sends SIGHUP and the default reaction is to just exit.
signal.signal(signal.SIGHUP, signal.SIG_IGN)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
if WAIT_FOR_BUTTON:
    GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


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
                time.sleep(0.05)

        cap = open_camera()
        if cap is None:
            return

        def _track_cap(new_cap):
            nonlocal cap
            cap = new_cap

        on_frame = None
        if SERVE_LIVE_VIEW:
            socketserver.ThreadingTCPServer.allow_reuse_address = True  # avoid "Address already in use" on a quick restart
            # If the port's genuinely held by something else (not just a
            # TIME_WAIT cooldown, which allow_reuse_address already
            # covers), try the next few ports rather than crashing the
            # whole run over a view that's a debugging aid, not core
            # functionality.
            port = LIVE_VIEW_PORT
            for attempt in range(5):
                try:
                    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), _StreamingHandler)
                    break
                except OSError as e:
                    print(f"Port {port} unavailable ({e}), trying {port + 1}...")
                    port += 1
            else:
                print("Could not bind a live-view port after 5 attempts - continuing without it.")
                server = None

            if server is not None:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                print(f"Live view at http://<pi-ip-address>:{port}/")
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
