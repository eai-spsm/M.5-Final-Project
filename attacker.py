import os
import time

import cv2
import numpy as np

from guidance import GuidedDrive
from perception import get_masks, cut_out

# Saves a snapshot of the ball mask (bitwise cutout) + its detected
# contour to disk every SNAPSHOT_INTERVAL_S, for reviewing detection
# quality after a run without needing a live view server. No cap on how
# many accumulate - clean out SNAPSHOT_DIR periodically by hand.
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_INTERVAL_S = 1.0

# Copied from goalkeeper.py and modified: same strafe-to-center-then-ram
# strategy, but the search (when the ball isn't visible at all) uses a
# rotate-based mode instead of goalkeeper.py's cord/strafe default - see
# SEARCH_MODE below. Same HSV perception as goalkeeper.py/main.py
# (perception/color_segment.py - no YOLO, no ArUco). Terminal-only status,
# no live view.

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Same idea as main.py's BALL_ROI_Y_START - exclude the background above
# the wall from ball detection. Brought back down from 0.60 - a snapshot
# showed 0.60 slicing straight through the real ball at moderate distance
# (only a thin sliver visible below the cutoff line), badly distorting its
# shape and likely failing the circularity filter even when it WAS in
# frame. That aggressive a cutoff isn't needed anymore anyway - the
# circularity/area filter (_find_ball) and the BALL_LOW/WALL_HIGH value-
# floor separation now handle the false-positive cases this was trying to
# paper over more precisely. PLACEHOLDER - tune against where the wall
# top/background boundary actually sits in frame.
BALL_ROI_Y_START = int(FRAME_HEIGHT * 0.40)

# Instead of a continuous angle (main.py's ball_angle_offset + rotate_to),
# split the frame into three zones - this matches how strafing actually
# works (a fixed-speed L/R burst, not a variable-rate turn), and is a much
# simpler decision than converting a pixel offset into a strafe
# distance/duration (which would need to know how far away the ball is,
# which nothing here measures).
# 55, not lower - movement/movement.py's WHEEL_TRIM gives FL/FR only 90%
# of whatever speed is commanded ({"fl": 0.9, "fr": 0.9, ...}), so at
# STRAFE_SPEED=45 the front wheels were only getting ~40.5% effective
# duty - below the ~45 "dead zone" floor documented in the README/
# CALIBRATION_REPORT, where a wheel doesn't have enough torque to
# actually turn. 55*0.9=49.5 clears it.
ZONE_LEFT_FRACTION = 1 / 3
ZONE_RIGHT_FRACTION = 2 / 3
STRAFE_SPEED = 55

# "Close" is read straight off the ROI: once the ball's centroid is this
# far down the frame, it's near enough to hit. PLACEHOLDER - tune against
# where the ball actually sits in frame at hitting distance.
CLOSE_Y_THRESHOLD = int(FRAME_HEIGHT * 0.75)

# If the ball isn't found for just a few frames while normally tracking
# (motion blur while strafing to center - strafing has no stop-and-look
# pause the way the rotate search does, brief occlusion), hold rather
# than immediately dropping into a full rotate search - a snapshot showed
# the ball cleanly centered one second, completely gone the next, then
# straight into "just spinning" instead of resuming tracking once it
# reappeared (which it almost certainly would have within a frame or two).
LOST_TRACK_GRACE_FRAMES = 5

# The launch rams forward until actual contact - the ball's mask covering
# CONTACT_AREA_FRACTION of the frame (genuinely close, not just near the
# bottom of frame) - with NO time limit, unlike goalkeeper.py: an attacker
# doesn't need to hold a defensive spot, so there's no reason to cap how
# far it commits to closing the distance. The wall-safety check (checked
# every step, before this) is still the backstop against actually running
# into something. After contact, it backs up for a fixed
# RETURN_ADJUST_DURATION_S just to re-open its view and re-track the ball,
# NOT a full retrace of however far the ram went (that's goalkeeper.py's
# job, holding a fixed position - this is an attacker, it wants to keep
# pushing forward overall, not undo its own progress).
LAUNCH_SPEED = 60
# The ball's mask covering at least this fraction of the frame is a much
# more reliable "actually close" signal than Y position alone - a ball
# that's still 50cm-1m away can fall out of the ROI/frame entirely just
# from the camera's downward FOV not reaching that far along the floor,
# which would get misread as "touching" if Y/vanishing alone were trusted.
# PLACEHOLDER - tune against the ball's actual apparent size at contact.
CONTACT_AREA_FRACTION = 0.12
RETURN_ADJUST_DURATION_S = 1.0  # fixed brief backup after contact, to adjust heading/re-see the ball - not a full return-to-start

# CLOSE_Y_THRESHOLD (where the launch triggers) and actual contact aren't
# far apart - if the ball's already close the instant the launch starts,
# or detection is just noisy for a frame, contact could register on
# literally the very next step. Without a floor on how long the forward
# phase must run, that cuts it down to a near-zero-duration blip - too
# brief for the motor's soft-start ramp (movement.py) to produce any real
# movement. This guarantees a real, visible push every time.
MIN_LAUNCH_DURATION_S = 0.3

# The ram has no time limit (see above) on the assumption it keeps aiming
# at a real, close ball - but if the ball goes off-camera mid-ram WITHOUT
# ever having been read close/large (rolled sideways, deflected, camera
# just lost it), that's not contact, and blindly ramming forward forever
# with nothing to aim at is wrong. LOST_ABORT_GRACE_S debounces a single-
# frame flicker (don't abort over one bad frame) but aborts fairly
# promptly - past that, cancel the ram and go back to searching instead
# of continuing to charge blind.
LOST_ABORT_GRACE_S = 0.3

# If the wall mask covers this much of the frame, back up regardless of
# what the ball's doing or mid-maneuver - basic collision safety, same as
# main.py.
WALL_COVERAGE_THRESHOLD = 0.80

# If the camera read fails this many times in a row, assume it's actually
# disconnected and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0

# Cord-based search (goalkeeper.py's default) - strafe toward
# +-SEARCH_STRAFE_CM (tracked X, not an angle), kept here as a selectable
# option even though it's not the default for this file (see SEARCH_MODE).
HOME_HEADING_DEG = 0.0
SEARCH_STRAFE_CM = 30
SEARCH_STRAFE_TIMEOUT_S = 3.0  # safety cap in case dead-reckoning drift means the target X is never quite reached
SEARCH_PAUSE_S = 0.4
STRAIGHT_HEADING_TOLERANCE_DEG = 5

# Rotate-based searches - see SEARCH_MODE below, this file's default.
# SEARCH_SWEEP_DEG (45) is the original blind jump-then-pause variant.
# ROTATE_SEARCH_SWEEP_DEG is the newer continuous-checking variant's
# target - a full turn (slightly under 360 to account for tracking
# imprecision) in one direction instead of a smaller side-to-side sweep,
# so nothing in the room is missed.
#
# ROTATE_SEARCH_SPEED must stay clear of the practical stall/dead-zone
# floor for rotation (same issue documented for strafe in the README -
# below a certain duty cycle a wheel doesn't have enough torque to
# actually turn at all). This was dropped to 25 to reduce motion blur
# over a full circle, but drive.pose()'s heading is PURE OPEN-LOOP dead
# reckoning (commanded speed x elapsed time, no real sensor feedback) -
# if the wheels stall at 25 and the robot doesn't actually turn, the
# tracked "swept degrees" in _rotate_sweep_side still climbs as if it
# did, so the search reports "completed a full sweep, nothing found"
# while barely having moved at all. goalkeeper.py's version of this never
# went below 45 for exactly this reason - matching that here.
#
# Rather than lowering ROTATE_SEARCH_SPEED further to reduce motion blur
# (which just risks the stall problem above again), this now rotates a
# small step at a time, comes to a FULL STOP, waits ROTATE_SEARCH_SETTLE_S,
# then captures - motion blur only happens while actually moving, so
# stopping before capturing eliminates it regardless of duty cycle. Slower
# overall per full sweep, but every frame it actually checks is sharp.
SEARCH_SWEEP_DEG = 45
ROTATE_SEARCH_SWEEP_DEG = 350
ROTATE_SEARCH_SPEED = 45
ROTATE_SEARCH_STEP_DEG = 20       # degrees per stop-and-look step - doubled from 10 to roughly halve full-sweep time, at the cost of a slightly higher chance of stepping past a narrow sighting of the ball between checks
ROTATE_SEARCH_SETTLE_S = 0.15    # pause after stopping, before capturing, so the frame isn't still catching up from the motion that just happened

# If the rotate search swings around to face the wall mid-sweep (nothing
# else was watching for it during the search's own blocking loop), back
# off briefly instead of continuing to spin toward/at it - see
# _rotate_sweep_side.
SEARCH_WALL_BACKUP_S = 0.3
SEARCH_WALL_BACKUP_SPEED = 55

# Which _search_for_ball_* variant attack_step uses when the ball isn't
# found - "cord" | "cord_straight" | "rotate_sweep" | "rotate_continuous".
# Switched to a rotate-based mode here (unlike goalkeeper.py's "cord"
# default) - all four are still kept side by side below so this is a
# one-line switch to compare them.
SEARCH_MODE = "rotate_continuous"

# Second ROI, separate from BALL_ROI_Y_START: a narrow band right at the
# bottom of the frame (closest to the camera) for motion detection - "is
# something closing in fast right in front of us". Cheap frame-differencing
# (grayscale absdiff), not YOLO - far less CPU/power, which matters given
# this Pi's already sitting close to an undervoltage threshold (see the
# soft-start fix in movement.py). Only meaningful while stationary -
# strafing/rotating shifts the whole scene, which would look exactly like
# something approaching.
NEAR_ROI_Y_START = int(FRAME_HEIGHT * 0.85)
MOTION_DIFF_THRESHOLD = 25       # grayscale intensity delta to count a pixel as "changed"
MOTION_PIXEL_FRACTION = 0.15     # fraction of the near-ROI that must change to call it motion

# Off for now: even gated to "only when HSV doesn't already see the ball"
# (see the call site), this kept intercepting real ball-tracking whenever
# HSV had a one-frame detection gap right as the ball approached - the
# ball's own motion is exactly what this was designed to notice, so a
# flicker at the wrong moment eats that step (no drive command issued)
# instead of letting the grace naturally resolve next frame. Not deleted -
# flip back to True if this is worth revisiting with a debounce.
NEAR_FIELD_MOTION_ENABLED = False


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


# Snapshots showed the real problem isn't WHERE in frame false positives
# show up (ROI alone can't fix it) - it's their SHAPE: thin, curved line
# artifacts (JPEG chroma-fringing along high-contrast edges elsewhere in
# the room) were getting picked up as "ball" just as often as the real
# round ball was. A real ball is a compact, roughly circular filled blob;
# these artifacts are long and thin. Local, stricter replacement for
# plain color_segment.ball_center() - same idea (largest qualifying
# contour's centroid) but also rejects anything too big or too irregular
# to plausibly be the ball. PLACEHOLDERS - tune against the ball's real
# apparent size/shape and how thin the actual noise contours are.
BALL_MAX_AREA_PX = 6000
BALL_MIN_CIRCULARITY = 0.55  # 4*pi*area/perimeter^2; 1.0 = perfect circle, thin lines score far lower

# At close range the ball itself can extend past the frame's own edges
# (not just the ROI cutoff - the same cropping problem, just at the
# camera's natural boundary instead), which distorts it into a
# crescent/chord shape the same way the ROI crop did - and that shape
# fails circularity too, right when it's closing in for real. Noise
# artifacts (thin lines) always had small area despite being long, so
# skip the circularity check entirely once a contour is already this
# large - at that size it's essentially never noise, more likely the
# ball itself, possibly edge-cropped. PLACEHOLDER - tune against the
# ball's real apparent area right as it starts touching a frame edge.
BALL_CIRCULARITY_BYPASS_AREA = 2000


def _find_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > BALL_MAX_AREA_PX:
            continue
        if area < BALL_CIRCULARITY_BYPASS_AREA:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < BALL_MIN_CIRCULARITY:
                continue
        if area > best_area:
            best, best_area = c, area
    if best is None:
        return None
    m = cv2.moments(best)
    if m["m00"] == 0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def _save_snapshot(frame, ball_mask, center):
    # The "final bitwise" - the ball mask cut out against the real frame
    # (same idea as color_segment.build_debug_view's ball cutout) - plus
    # the actual contour ball_center() found it from, drawn on top, so a
    # saved snapshot shows exactly what the detector saw and picked.
    view = cut_out(frame, ball_mask)
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(view, [largest], -1, (0, 0, 255), 2)
    if center is not None:
        cv2.circle(view, center, 6, (0, 255, 255), 2)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"ball_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg")
    cv2.imwrite(path, view)


def _maybe_save_snapshot(search_state, frame, ball_mask, center):
    # Time-gated wrapper so this can be called both from attack_step AND
    # from inside a blocking search sweep - the search functions run as
    # one long call from attack_step's point of view, so without this
    # inside the sweep loop too, zero snapshots would be taken for the
    # entire duration of a rotate search (which is exactly what was
    # happening - the saved snapshots had multi-second/tens-of-second
    # gaps whenever a search ran, since nothing was watching during it).
    now = time.time()
    if now - search_state.get("last_snapshot_time", 0) >= SNAPSHOT_INTERVAL_S:
        search_state["last_snapshot_time"] = now
        _save_snapshot(frame, ball_mask, center)


def _near_field_motion(frame, search_state):
    # Frame-differencing within the near-field ROI only - returns True if
    # enough of that band changed since the last check to call it
    # something approaching. Gated to when the bot is stationary; resets
    # its reference frame whenever the bot starts moving so it doesn't
    # compare across a strafe/launch and falsely trigger the instant it
    # stops again.
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


def _strafe_until(cap, drive, target_x, move):
    # Blocking: strafes in short bursts (move = "strafe_left" or
    # "strafe_right"), checking the tracked X (Navigator's dead reckoning
    # - cord, not an angle/heading) and the ball after every burst. Stops
    # and returns the instant either the ball's spotted or target_x is
    # reached, whichever comes first. SEARCH_STRAFE_TIMEOUT_S is a safety
    # cap in case dead-reckoning drift means target_x never quite gets
    # hit.
    deadline = time.time() + SEARCH_STRAFE_TIMEOUT_S
    while time.time() < deadline:
        x = drive.pose()[0]
        reached = x <= target_x if move == "strafe_left" else x >= target_x
        if reached:
            break
        getattr(drive, move)(speed=STRAFE_SPEED)
        time.sleep(SEARCH_PAUSE_S)
        drive.stop()
        ret, frame = cap.read()
        if ret:
            masks = get_masks(frame)
            center = _find_ball(_ball_roi(masks["ball"]))
            if center is not None:
                return frame, masks, center
    drive.stop()
    return None, None, None


def _search_for_ball(cap, drive, search_state=None):
    # Ball's not visible - strafe toward +-SEARCH_STRAFE_CM (tracked X,
    # cord not angle) checking for it after every short burst. SEARCH_MODE
    # = "cord" to use this one.
    start_x = drive.pose()[0]

    frame, masks, center = _strafe_until(cap, drive, start_x - SEARCH_STRAFE_CM, "strafe_left")
    if center is not None:
        return frame, masks, center

    _strafe_until(cap, drive, start_x, "strafe_right")

    frame, masks, center = _strafe_until(cap, drive, start_x + SEARCH_STRAFE_CM, "strafe_right")
    if center is not None:
        return frame, masks, center

    current_x = drive.pose()[0]
    _strafe_until(cap, drive, start_x, "strafe_left" if current_x > start_x else "strafe_right")
    return None, None, None


def _strafe_straight_until(cap, drive, target_x, move):
    # Same idea as _strafe_until(), plus a heading check/correction after
    # every burst. SEARCH_MODE = "cord_straight".
    deadline = time.time() + SEARCH_STRAFE_TIMEOUT_S
    while time.time() < deadline:
        x = drive.pose()[0]
        reached = x <= target_x if move == "strafe_left" else x >= target_x
        if reached:
            break
        getattr(drive, move)(speed=STRAFE_SPEED)
        time.sleep(SEARCH_PAUSE_S)
        drive.stop()

        heading = drive.pose()[2]
        heading_diff = (heading - HOME_HEADING_DEG + 180) % 360 - 180  # -180..180
        if abs(heading_diff) > STRAIGHT_HEADING_TOLERANCE_DEG:
            drive.rotate_to(HOME_HEADING_DEG)

        ret, frame = cap.read()
        if ret:
            masks = get_masks(frame)
            center = _find_ball(_ball_roi(masks["ball"]))
            if center is not None:
                return frame, masks, center
    drive.stop()
    return None, None, None


def _search_for_ball_straight(cap, drive, search_state=None):
    # Alternate cord-based search with heading-drift correction.
    # SEARCH_MODE = "cord_straight" to use this one.
    start_x = drive.pose()[0]

    frame, masks, center = _strafe_straight_until(cap, drive, start_x - SEARCH_STRAFE_CM, "strafe_left")
    if center is not None:
        return frame, masks, center

    _strafe_straight_until(cap, drive, start_x, "strafe_right")

    frame, masks, center = _strafe_straight_until(cap, drive, start_x + SEARCH_STRAFE_CM, "strafe_right")
    if center is not None:
        return frame, masks, center

    current_x = drive.pose()[0]
    _strafe_straight_until(cap, drive, start_x, "strafe_left" if current_x > start_x else "strafe_right")
    return None, None, None


def _search_for_ball_rotate_sweep(cap, drive, search_state=None):
    # Blind rotate_to() calls block until each turn completes (or times
    # out): look left, pause and check, look right, pause and check.
    # Returns as soon as either side spots something. Always rotates back
    # to HOME_HEADING_DEG before returning - whether or not anything was
    # found - then takes one more fresh frame there so the L/C/R zone read
    # afterward is relative to the home-facing view, not whatever angle it
    # was found at. SEARCH_MODE = "rotate_sweep" to use this one.
    found = False
    for offset in (-SEARCH_SWEEP_DEG, SEARCH_SWEEP_DEG):
        drive.rotate_to((HOME_HEADING_DEG + offset) % 360)
        time.sleep(SEARCH_PAUSE_S)
        ret, frame = cap.read()
        if ret and _find_ball(_ball_roi(get_masks(frame)["ball"])) is not None:
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
    return frame, masks, _find_ball(_ball_roi(masks["ball"]))


def _rotate_sweep_side(cap, drive, move, search_state):
    # Rotates a small step (ROTATE_SEARCH_STEP_DEG), comes to a FULL STOP,
    # waits ROTATE_SEARCH_SETTLE_S, then captures and checks - repeated up
    # to ROTATE_SEARCH_SWEEP_DEG total, stopping the instant the ball's
    # spotted. Capturing only once fully stopped (rather than mid-rotation)
    # means every checked frame is sharp regardless of ROTATE_SEARCH_SPEED
    # - see the comment above that constant for why slowing the rotation
    # itself risks a different problem (stalling). Also checks wall
    # coverage every step (the search loop reads frames independently of
    # the outer attack_step, so nothing else was watching for a wall
    # mid-sweep) - if it swings around to face the wall, back off
    # immediately instead of continuing to spin at/into it. Returns
    # "found" / "wall" / "none".
    #
    # Also saves a snapshot every step (time-gated, same as attack_step) -
    # a full sweep can take several seconds as ONE call from attack_step's
    # point of view, so without this nothing gets saved for the entire
    # duration of a search; that blind spot was exactly why "can't detect
    # during the 360 turn" had no visual evidence to diagnose from.
    step_sign = 1 if move == "rotate_right" else -1
    swept = 0.0
    heading = drive.pose()[2]
    while swept < ROTATE_SEARCH_SWEEP_DEG:
        heading = (heading + step_sign * ROTATE_SEARCH_STEP_DEG) % 360
        drive.rotate_to(heading, speed=ROTATE_SEARCH_SPEED)
        time.sleep(ROTATE_SEARCH_SETTLE_S)
        swept += ROTATE_SEARCH_STEP_DEG

        ret, frame = cap.read()
        if not ret:
            continue
        masks = get_masks(frame)
        ball_mask = _ball_roi(masks["ball"])
        center = _find_ball(ball_mask)
        _maybe_save_snapshot(search_state, frame, ball_mask, center)

        wall_fraction = float((masks["wall"] > 0).mean())
        if wall_fraction >= WALL_COVERAGE_THRESHOLD:
            drive.stop()
            drive.backward(speed=SEARCH_WALL_BACKUP_SPEED)
            time.sleep(SEARCH_WALL_BACKUP_S)
            drive.stop()
            return "wall"
        if center is not None:
            return "found"
    return "none"


def _search_for_ball_rotate_continuous(cap, drive, search_state):
    # This file's default search (see SEARCH_MODE): one continuous slow
    # turn (small steps, checking every step - no discrete jump-then-
    # pause) covering a full ROTATE_SEARCH_SWEEP_DEG (~360) in a single
    # direction, instead of a smaller side-to-side sweep - covers
    # everything around it rather than just +-90 either side of where it
    # started. Stops the instant the ball shows up and stays facing it -
    # only returns to the heading it started at when NOTHING was found
    # (a real find used to get overridden by an unconditional fast
    # rotate_to(home_heading) right after - that swung the camera off the
    # ball it had just found, at default speed rather than the deliberate
    # slow ROTATE_SEARCH_SPEED, before the "final" frame was even taken).
    home_heading = drive.pose()[2]
    result = _rotate_sweep_side(cap, drive, "rotate_right", search_state)
    if result == "wall":
        # Already backed off inside _rotate_sweep_side - abandon this
        # search attempt rather than continuing to spin near the wall;
        # the outer loop re-evaluates fresh next step.
        drive.rotate_to(home_heading)
        return None, None, None

    drive.stop()
    if result != "found":
        drive.rotate_to(home_heading)
        return None, None, None

    time.sleep(ROTATE_SEARCH_SETTLE_S)
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    masks = get_masks(frame)
    return frame, masks, _find_ball(_ball_roi(masks["ball"]))


_SEARCH_MODES = {
    "cord": _search_for_ball,
    "cord_straight": _search_for_ball_straight,
    "rotate_sweep": _search_for_ball_rotate_sweep,
    "rotate_continuous": _search_for_ball_rotate_continuous,
}


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


def attack_step(cap, drive, search_state=None):
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    ball_mask = _ball_roi(masks["ball"])
    center = _find_ball(ball_mask)

    _maybe_save_snapshot(search_state, frame, ball_mask, center)

    wall_fraction = float((masks["wall"] > 0).mean())
    if wall_fraction >= WALL_COVERAGE_THRESHOLD:
        search_state.pop("launch_phase", None)
        if _safe_move(drive, search_state, "backward"):
            return f"Wall fills {wall_fraction * 100:.0f}% of view - backing up"
        return f"Wall fills {wall_fraction * 100:.0f}% of view - braking before backing up"

    # Only fires while stationary (see _near_field_motion), and only when
    # HSV doesn't already see the ball - this returns early with no drive
    # command at all, so if the ball itself was causing the motion (very
    # likely - it's the main thing moving in view while tracking it), this
    # was permanently blocking the zone/launch logic below from ever
    # running once triggered: the robot just sits there repeating "motion
    # detected" every step since nothing else ever issues a new command.
    # Real HSV ball detection is more specific/reliable than a generic
    # motion blip and should always take priority when both fire.
    if NEAR_FIELD_MOTION_ENABLED and center is None and _near_field_motion(frame, search_state):
        return "Motion detected close (near-field) - incoming object"

    # Continue an in-progress launch-and-return maneuver before anything
    # below gets a chance to interrupt it early with a fresh zone/distance
    # read.
    launch_phase = search_state.get("launch_phase")
    if launch_phase is not None:
        elapsed = time.time() - search_state["launch_start"]
        if launch_phase == "forward":
            ball_area = int((ball_mask > 0).sum())
            frame_area = frame.shape[0] * frame.shape[1]
            large_enough = ball_area >= CONTACT_AREA_FRACTION * frame_area
            if center is not None and large_enough:
                search_state["launch_max_area_seen"] = True
            contact = (center is not None and large_enough) or (
                center is None and search_state.get("launch_max_area_seen", False)
            )

            # Ball genuinely gone (not a contact vanish, not just a
            # one-frame flicker) - stop charging blind and go back to
            # searching instead.
            if center is None and not search_state.get("launch_max_area_seen", False):
                lost_since = search_state.get("launch_lost_since")
                if lost_since is None:
                    search_state["launch_lost_since"] = time.time()
                elif time.time() - lost_since >= LOST_ABORT_GRACE_S:
                    search_state.pop("launch_phase", None)
                    search_state.pop("launch_start", None)
                    search_state.pop("launch_max_area_seen", None)
                    search_state.pop("launch_lost_since", None)
                    _stop(drive, search_state)
                    return "Ball lost mid-ram - aborting, back to search"
            else:
                search_state.pop("launch_lost_since", None)

            if elapsed < MIN_LAUNCH_DURATION_S or not contact:
                _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
                return f"Ramming forward ({elapsed:.1f}s)"
            search_state["launch_phase"] = "returning"
            search_state["launch_start"] = time.time()
            elapsed = 0.0
        if elapsed < RETURN_ADJUST_DURATION_S:
            if _safe_move(drive, search_state, "backward", LAUNCH_SPEED):
                return f"Adjusting heading ({elapsed:.1f}s)"
            return f"Braking before adjusting ({elapsed:.1f}s)"
        search_state.pop("launch_phase", None)
        search_state.pop("launch_start", None)
        search_state.pop("launch_max_area_seen", None)
        return "Launch complete - resuming attack"

    if center is None:
        # A single missed frame (motion blur while strafing to center,
        # brief occlusion) shouldn't immediately dump into a full rotate
        # search - hold for a few frames first and let normal tracking
        # resume once it reappears, which it very likely will if this was
        # just a blur/flicker rather than a genuine loss.
        miss_streak = search_state.get("track_miss_streak", 0) + 1
        search_state["track_miss_streak"] = miss_streak
        if miss_streak <= LOST_TRACK_GRACE_FRAMES and search_state.get("ever_tracked"):
            _stop(drive, search_state)
            return f"Ball flicker ({miss_streak}/{LOST_TRACK_GRACE_FRAMES}) - holding"

        frame, masks, center = _SEARCH_MODES[SEARCH_MODE](cap, drive, search_state)
        search_state["last_move"] = "stop"  # every _search_for_ball_* variant always ends stopped
        if center is None:
            _stop(drive, search_state)
            return "No ball found (rotate search) - holding position"

    search_state["ever_tracked"] = True
    search_state["track_miss_streak"] = 0
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
        search_state["launch_max_area_seen"] = False
        _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
        return f"Ball det C {coord} - close, launching forward"

    _stop(drive, search_state)
    return f"Ball det C {coord} - holding (not close)"


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def attack_loop(cap, drive, on_reconnect=None):
    consecutive_failures = 0
    search_state = {}
    while True:
        start = time.time()
        status = attack_step(cap, drive, search_state=search_state)
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
    print("Attacker demo. Strafes to keep the ball centered, launches forward and back once close.")
    print(f"Search mode: {SEARCH_MODE}. Ctrl+C to stop.")
    try:
        attack_loop(cap, drive, on_reconnect=_track_cap)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        drive.stop()
        drive.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
