import signal
import time

import cv2
import RPi.GPIO as GPIO

from guidance import GuidedDrive
from perception import get_masks, find_goal_gap, ball_angle_offset, ball_center

# Single entry point: camera + perception + drive state machine, all in
# one file. Perception is deliberately just HSV color
# (perception/color_segment.py: get_masks/ball_center/ball_angle_offset/
# find_goal_gap) - no YOLO, no ArUco, no separate detector modules. That
# color-only approach is what's actually been working well; everything
# else tried on top of it added latency/complexity without a clear win.
# Terminal-only status output, same as goalkeeper.py - no live-view HTTP
# server (that's real per-step overhead: JPEG-encoding and serving a
# frame every loop iteration), use cam_control.py separately if a live
# view is needed for tuning/aiming.

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Berserk attacker: rotate (not strafe) to face the ball, faster than a
# cautious turn, and charge forward the moment it's roughly pointed at it
# rather than waiting to precisely center first. If it sees green, it
# goes for it. Priority order, checked every step:
#   1. Recover from being pinned against a wall (overrides everything -
#      no point charging harder into whatever it's already stuck on)
#   2. Don't go into the goal
#   3. Evade an incoming rear threat
#   4. Find/charge the ball (default behavior)

# Looser than a cautious approach - charge forward once roughly pointed
# at the ball instead of precisely centering first.
CENTERED_TOLERANCE_DEG = 15

# Faster than a cautious turn/approach (movement.py's default duty cycle
# is 60) - snap toward the ball and close distance aggressively.
ROTATE_SPEED = 75
CHARGE_SPEED = 75

# Duty cycle % while spinning to search when the ball isn't visible at
# all - slower than ROTATE_SPEED/CHARGE_SPEED so a frame doesn't blur
# past the ball and miss it while blindly spinning with nothing to lock
# onto yet.
SEARCH_SPEED = 35

# The wall is low enough that background well outside the field (chairs,
# bags, anything else green) is visible above it in frame - that's most
# of the "noise" HSV was picking up. Exclude everything above this row
# from ball detection entirely (see _ball_roi). PLACEHOLDER - tune
# against where the wall top/background boundary actually sits in frame.
BALL_ROI_Y_START = int(FRAME_HEIGHT * 0.35)

# If the wall mask covers this much of the frame, treat it as pinned
# against the wall - see the pin-recovery comment below.
WALL_COVERAGE_THRESHOLD = 0.80

# Pinned against the wall (stuck pushing straight into it) - back off
# briefly, then strafe toward whichever side the ball was last seen on
# instead of just backing straight out and re-charging the same straight
# line into it again. Approaching from an angle after backing off is more
# likely to actually clear whatever it's pinned against.
PIN_BACKUP_S = 0.3
PIN_STRAFE_S = 0.5
PIN_STRAFE_SPEED = 55

# The ultrasonic sensor is mounted at the REAR - it watches for something
# closing in from behind, not a front obstruction. It can't say which
# side a rear threat is on, and strafing wouldn't increase distance from
# something behind us anyway - push forward instead, which does.
REAR_THREAT_DISTANCE_CM = 15
EVADE_SPEED = 60

# How wide the detected goal gap needs to be (as a fraction of frame width)
# before treating it as "close enough to stop" - matches how big the
# opening looks right before the goal mouth.
GOAL_GAP_CLOSE_WIDTH_FRACTION = 0.5

# If the camera read fails this many times in a row, assume it's actually
# disconnected (not just a one-off dropped frame) and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21
WAIT_FOR_BUTTON = False  # no button wired up right now - set True once it is


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
_OPPOSITES = {
    "forward": "backward", "backward": "forward",
    "strafe_left": "strafe_right", "strafe_right": "strafe_left",
}


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


def chase_step(cap, drive, search_state=None):
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    center = ball_center(_ball_roi(masks["ball"]))
    if center is not None:
        # Remembered even mid-pin/evade so the strafe-from-an-angle
        # recovery below knows which side to go around toward.
        search_state["last_ball_angle"] = ball_angle_offset(center[0], frame.shape[1])

    # --- Priority 1: recover from being pinned against a wall ----------
    # Instead of just backing straight out and re-charging the same
    # straight line into it, back off briefly then strafe toward
    # whichever side the ball was last seen on - approaching from an
    # angle is more likely to actually clear whatever it's stuck on.
    pin_phase = search_state.get("pin_phase")
    if pin_phase is not None:
        elapsed = time.time() - search_state["pin_start"]
        if pin_phase == "backward":
            if elapsed < PIN_BACKUP_S:
                _safe_move(drive, search_state, "backward")
                return f"Pinned - backing off ({elapsed:.1f}s)"
            search_state["pin_phase"] = "strafe"
            search_state["pin_start"] = time.time()
            elapsed = 0.0
        if elapsed < PIN_STRAFE_S:
            move = "strafe_left" if search_state.get("last_ball_angle", 0) < 0 else "strafe_right"
            if _safe_move(drive, search_state, move, PIN_STRAFE_SPEED):
                return f"Pinned - repositioning from the side ({elapsed:.1f}s)"
            return f"Pinned - braking before repositioning ({elapsed:.1f}s)"
        search_state.pop("pin_phase", None)
        search_state.pop("pin_start", None)
        return "Unpinned - resuming attack"

    wall_fraction = float((masks["wall"] > 0).mean())
    if wall_fraction >= WALL_COVERAGE_THRESHOLD:
        search_state["pin_phase"] = "backward"
        search_state["pin_start"] = time.time()
        if _safe_move(drive, search_state, "backward"):
            return f"Wall fills {wall_fraction * 100:.0f}% of view - pinned, backing off"
        return f"Wall fills {wall_fraction * 100:.0f}% of view - braking before backing off"

    # --- Priority 2: don't go into the goal -----------------------------
    gap = find_goal_gap(masks["wall"])
    close_by_width = gap is not None and gap["width_px"] >= GOAL_GAP_CLOSE_WIDTH_FRACTION * frame.shape[1]
    if close_by_width:
        if _safe_move(drive, search_state, "backward"):
            return "Near goal - backing out"
        return "Near goal - braking before backing out"

    if gap is not None and center is not None:
        gap_left = gap["center_x"] - gap["width_px"] / 2
        gap_right = gap["center_x"] + gap["width_px"] / 2
        if gap_left <= center[0] <= gap_right:
            # The ball itself is sitting inside the goal opening - don't
            # chase it in.
            if _safe_move(drive, search_state, "backward"):
                return "Ball is in the goal opening - backing out"
            return "Ball is in the goal opening - braking before backing out"

    # --- Priority 3: evade an incoming rear threat ----------------------
    distance = drive.get_distance()
    rear_threat = distance is not None and distance < REAR_THREAT_DISTANCE_CM
    if rear_threat:
        # Rear-facing ultrasonic - something's closing in from behind. It
        # can't say which side, and strafing wouldn't increase distance
        # from a rear threat anyway - push forward instead, which does.
        if _safe_move(drive, search_state, "forward", EVADE_SPEED):
            return f"Rear threat at {distance:.0f}cm - pushing forward"
        return f"Rear threat at {distance:.0f}cm - braking before pushing forward"

    # --- Priority 4: go berserk on the ball ------------------------------
    if center is None:
        # No green in view at all - spin fast until some shows up.
        drive.rotate_right(speed=SEARCH_SPEED)
        search_state["last_move"] = "rotate"  # not forward/backward - clears stale linear-move tracking
        return "Searching (spinning fast)..."

    angle = ball_angle_offset(center[0], frame.shape[1])
    if abs(angle) > CENTERED_TOLERANCE_DEG:
        # Snap toward it - continuous fast rotation each step rather than
        # a slow, precise blocking turn.
        if angle > 0:
            drive.rotate_right(speed=ROTATE_SPEED)
        else:
            drive.rotate_left(speed=ROTATE_SPEED)
        search_state["last_move"] = "rotate"
        return f"Ball at {angle:+5.1f} deg - snapping toward it"

    if _safe_move(drive, search_state, "forward", CHARGE_SPEED):
        return f"Ball roughly ahead ({angle:+5.1f} deg) - CHARGE"
    return f"Ball roughly ahead ({angle:+5.1f} deg) - braking before charge"


def _safe_print(*args, **kwargs):
    # If stdout is gone (e.g. an SSH session dropped), printing raises
    # BrokenPipeError/OSError - that's fine to ignore, the driving logic
    # doesn't need anyone watching to keep working.
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def chase_loop(cap, drive, on_reconnect=None):
    consecutive_failures = 0
    search_state = {}
    while True:
        start = time.time()
        status = chase_step(cap, drive, search_state=search_state)
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

        print("Berserk attacker. Rotates fast to face the ball and charges. Ctrl+C to stop.")
        chase_loop(cap, drive, on_reconnect=_track_cap)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    finally:
        print("Cleaning up GPIO resources...")
        drive.stop()
        drive.cleanup()
        if cap is not None:
            cap.release()
        print("Done!")


if __name__ == "__main__":
    main()
