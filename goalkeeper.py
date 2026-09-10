import time

import cv2

from guidance import GuidedDrive
from perception import get_masks, ball_center

# Alternate strategy to main.py's chase-and-approach: instead of turning
# to face the ball and driving straight at it, hold position (facing one
# fixed direction, e.g. out toward the field) and strafe left/right to
# keep the ball centered, then punch forward to hit/launch it back once
# it's close, and return to the starting spot afterward instead of
# drifting forward with the launch. Same HSV perception as main.py
# (perception/color_segment.py - no YOLO, no ArUco), just a different
# decision layer on top of it. Terminal-only status, no live view.

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Same idea as main.py's BALL_ROI_Y_START - exclude the background above
# the wall from ball detection. PLACEHOLDER - tune against where the wall
# top/background boundary actually sits in frame.
BALL_ROI_Y_START = int(FRAME_HEIGHT * 0.35)

# Instead of a continuous angle (main.py's ball_angle_offset + rotate_to),
# split the frame into three zones - this matches how strafing actually
# works (a fixed-speed L/R burst, not a variable-rate turn), and is a much
# simpler decision than converting a pixel offset into a strafe
# distance/duration (which would need to know how far away the ball is,
# which nothing here measures).
ZONE_LEFT_FRACTION = 1 / 3
ZONE_RIGHT_FRACTION = 2 / 3
STRAFE_SPEED = 45

# "Close" is read straight off the ROI: once the ball's centroid is this
# far down the frame, it's near enough to hit. PLACEHOLDER - tune against
# where the ball actually sits in frame at hitting distance.
CLOSE_Y_THRESHOLD = int(FRAME_HEIGHT * 0.75)

# The launch is time-boxed and symmetric: drive forward for
# LAUNCH_DURATION_S to hit the ball, then drive backward for the same
# duration to undo that move - keeps the bot near its starting spot
# instead of drifting forward on every hit, without needing real
# position tracking (Navigator's dead reckoning isn't calibrated - see
# guidance/guided_drive.py). PLACEHOLDER - tune against how far forward
# LAUNCH_DURATION_S at LAUNCH_SPEED actually covers.
LAUNCH_SPEED = 60
LAUNCH_DURATION_S = 0.4

# If the wall mask covers this much of the frame, back up regardless of
# what the ball's doing or mid-maneuver - basic collision safety, same as
# main.py.
WALL_COVERAGE_THRESHOLD = 0.80

# If the camera read fails this many times in a row, assume it's actually
# disconnected and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0

# The whole point of this strategy is staying facing outward and using
# strafes (not turns) to track/move - so when the ball isn't visible, the
# search is a brief look-around GESTURE, not a heading change: rotate to
# HOME_HEADING_DEG - SEARCH_SWEEP_DEG, pause and check, then
# +SEARCH_SWEEP_DEG, pause and check, then ALWAYS rotate back to
# HOME_HEADING_DEG afterward regardless of whether anything was found -
# unlike main.py's continuous spin search, this bot must not end up
# facing some other direction. PLACEHOLDER - tune sweep angle/pause
# against how wide the camera's FOV actually is.
HOME_HEADING_DEG = 0.0
SEARCH_SWEEP_DEG = 45
SEARCH_PAUSE_S = 0.4

# Second ROI, separate from BALL_ROI_Y_START: a narrow band right at the
# bottom of the frame (closest to the camera) for motion detection - "is
# something closing in fast right in front of us" (opponent car, or the
# ball moving quickly), as opposed to BALL_ROI_Y_START's whole-floor band
# used for finding the ball's position. Cheap frame-differencing
# (grayscale absdiff), not YOLO - far less CPU/power, which matters given
# this Pi's already sitting close to an undervoltage threshold (see the
# soft-start fix in movement.py).
#
# Only meaningful while the bot itself is stationary: strafing/rotating
# shifts the whole scene including this band, which would look exactly
# like something approaching. Gated on search_state["last_move"] == "stop"
# rather than trying to compensate for the bot's own motion.
NEAR_ROI_Y_START = int(FRAME_HEIGHT * 0.85)
MOTION_DIFF_THRESHOLD = 25       # grayscale intensity delta to count a pixel as "changed"
MOTION_PIXEL_FRACTION = 0.15     # fraction of the near-ROI that must change to call it motion


# movement/movement.py's set_wheels() flips the H-bridge direction pins
# directly with no braking step - reversing straight from one direction to
# its exact opposite (forward<->backward, strafe_left<->strafe_right) at
# full duty cycle spikes current/back-EMF across all 4 motors at once hard
# enough to brown out the Pi's power rail, which can reset the Wi-Fi chip
# (or the whole Pi) and looks exactly like a dropped SSH session. _safe_move
# forces one coast-to-a-stop step before any such reversal instead of
# flipping directly - costs at most one extra ~step's worth of delay.
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
    roi = mask.copy()
    roi[:BALL_ROI_Y_START, :] = 0
    return roi


def _near_field_motion(frame, search_state):
    # Frame-differencing within the near-field ROI only - returns True if
    # enough of that band changed since the last check to call it
    # something approaching. Gated to when the bot is stationary (see the
    # NEAR_ROI comment above); resets its reference frame whenever the bot
    # starts moving so it doesn't compare across a strafe/launch and
    # falsely trigger the instant it stops again.
    if search_state.get("last_move") != "stop":
        search_state.pop("near_field_prev", None)
        return False

    near = frame[NEAR_ROI_Y_START:, :]
    gray = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
    prev = search_state.get("near_field_prev")
    search_state["near_field_prev"] = gray
    if prev is None or prev.shape != gray.shape:
        return False

    diff = cv2.absdiff(gray, prev)
    changed_fraction = float((diff > MOTION_DIFF_THRESHOLD).mean())
    return changed_fraction >= MOTION_PIXEL_FRACTION


def _ball_zone(center_x, frame_width):
    if center_x < frame_width * ZONE_LEFT_FRACTION:
        return "L"
    if center_x > frame_width * ZONE_RIGHT_FRACTION:
        return "R"
    return "C"


def _search_for_ball(cap, drive):
    # Blind rotate_to() calls block until each turn completes (or times
    # out), so this plays out as a genuine gesture: look left, pause and
    # check, look right, pause and check. Returns as soon as either side
    # spots something. Always rotates back to HOME_HEADING_DEG before
    # returning - whether or not anything was found - then takes one more
    # fresh frame there so the L/C/R zone read afterward is relative to
    # the home-facing view, not whatever angle it was found at.
    found = False
    for offset in (-SEARCH_SWEEP_DEG, SEARCH_SWEEP_DEG):
        drive.rotate_to((HOME_HEADING_DEG + offset) % 360)
        time.sleep(SEARCH_PAUSE_S)
        ret, frame = cap.read()
        if ret and ball_center(_ball_roi(get_masks(frame)["ball"])) is not None:
            found = True
            break

    drive.rotate_to(HOME_HEADING_DEG)
    if not found:
        return None, None, None

    time.sleep(SEARCH_PAUSE_S)
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    masks = get_masks(frame)
    return frame, masks, ball_center(_ball_roi(masks["ball"]))


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


def defend_step(cap, drive, search_state=None):
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    ball_mask = _ball_roi(masks["ball"])

    wall_fraction = float((masks["wall"] > 0).mean())
    if wall_fraction >= WALL_COVERAGE_THRESHOLD:
        search_state.pop("launch_phase", None)
        if _safe_move(drive, search_state, "backward"):
            return f"Wall fills {wall_fraction * 100:.0f}% of view - backing up"
        return f"Wall fills {wall_fraction * 100:.0f}% of view - braking before backing up"

    # Only fires while stationary (see _near_field_motion) - a launch or
    # strafe already in progress isn't interrupted by this.
    if _near_field_motion(frame, search_state):
        return "Motion detected close (near-field) - incoming object"

    # Continue an in-progress launch-and-return maneuver before anything
    # below gets a chance to interrupt it early with a fresh zone/distance
    # read.
    launch_phase = search_state.get("launch_phase")
    if launch_phase is not None:
        elapsed = time.time() - search_state["launch_start"]
        if launch_phase == "forward":
            if elapsed < LAUNCH_DURATION_S:
                _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
                return f"Launching forward ({elapsed:.1f}s)"
            search_state["launch_phase"] = "returning"
            search_state["launch_start"] = time.time()
            elapsed = 0.0
        if elapsed < LAUNCH_DURATION_S:
            if _safe_move(drive, search_state, "backward", LAUNCH_SPEED):
                return f"Returning to position ({elapsed:.1f}s)"
            return f"Braking before returning ({elapsed:.1f}s)"
        search_state.pop("launch_phase", None)
        search_state.pop("launch_start", None)
        return "Launch complete - resuming defense"

    center = ball_center(ball_mask)
    if center is None:
        frame, masks, center = _search_for_ball(cap, drive)
        search_state["last_move"] = "stop"  # rotate_to() always ends stopped
        if center is None:
            _stop(drive, search_state)
            return "No ball found (looked L/R) - holding at home heading"

    x, y = center
    zone = _ball_zone(x, frame.shape[1])
    coord = f"(x={x}, y={y})"

    if zone == "L":
        if _safe_move(drive, search_state, "strafe_left", STRAFE_SPEED):
            return f"Ball det L {coord} - moving to C (strafe left)"
        return f"Ball det L {coord} - braking before reversing to strafe left"
    if zone == "R":
        if _safe_move(drive, search_state, "strafe_right", STRAFE_SPEED):
            return f"Ball det R {coord} - moving to C (strafe right)"
        return f"Ball det R {coord} - braking before reversing to strafe right"

    # Centered - launch once it's close enough (read straight off the ROI:
    # the ball's y-position in frame), otherwise just hold.
    if y >= CLOSE_Y_THRESHOLD:
        search_state["launch_phase"] = "forward"
        search_state["launch_start"] = time.time()
        _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
        return f"Ball det C {coord} - close, launching forward"

    _stop(drive, search_state)
    return f"Ball det C {coord} - holding (not close)"


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def defend_loop(cap, drive, on_reconnect=None):
    consecutive_failures = 0
    search_state = {}
    while True:
        start = time.time()
        status = defend_step(cap, drive, search_state=search_state)
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

        _safe_print(f"\r{status} [{elapsed_ms:.0f}ms/step]{'':<20}", end="", flush=True)


def main():
    cap = open_camera()
    if cap is None:
        return

    def _track_cap(new_cap):
        nonlocal cap
        cap = new_cap

    drive = GuidedDrive()
    print("Goalkeeper demo. Strafes to keep the ball centered, launches forward and back once it's close.")
    print("Ctrl+C to stop.")
    try:
        defend_loop(cap, drive, on_reconnect=_track_cap)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        drive.stop()
        drive.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
