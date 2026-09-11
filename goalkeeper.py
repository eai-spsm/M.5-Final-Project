import time

import cv2
import numpy as np

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
# Not lower than ~55 - movement/movement.py's WHEEL_TRIM gives FL/FR only
# 90% of whatever speed is commanded ({"fl": 0.9, "fr": 0.9, ...}), so at
# STRAFE_SPEED=45 the front wheels were only getting ~40.5% effective
# duty - below the ~45 "dead zone" floor documented in the README/
# CALIBRATION_REPORT, where a wheel doesn't have enough torque to
# actually turn. That's almost certainly what "only 2 back wheels move"
# during strafe was - RL/RR (trim 1.0) got the full duty and spun, FL/FR
# didn't clear the stall threshold.
#
# Bumped again (60 -> 65), deliberately PAST the ~60 effective-duty
# ceiling documented in CALIBRATION_REPORT.md as where the same-board-
# opposite-channel glitch (a wheel visibly flipping direction mid-hold)
# starts reappearing - strafing is the one command that drives both
# channels on the SAME board in OPPOSITE directions (forward/backward
# always keep a board's two channels in sync, so they don't have this
# risk). RL/RR (trim 1.0) now sit at 65 effective duty, 5 over that
# ceiling - watch for the glitch on real hardware; if it reappears, that's
# confirmation this needs to come back down, not a surprise.
ZONE_LEFT_FRACTION = 1 / 3
ZONE_RIGHT_FRACTION = 2 / 3
STRAFE_SPEED = 65

# "Close" is read straight off the ROI: once the ball's centroid is this
# far down the frame, it's near enough to hit. PLACEHOLDER - tune against
# where the ball actually sits in frame at hitting distance.
CLOSE_Y_THRESHOLD = int(FRAME_HEIGHT * 0.75)

# The launch rams forward until actual contact - the ball dropping below
# view (under/against the camera, i.e. touching) or reaching
# CONTACT_Y_THRESHOLD - then drives backward for exactly as long as the
# ram took, to undo that move and return to the starting spot instead of
# drifting forward on every hit. Time-symmetric rather than distance-based
# since there's no reliable real position tracking (Navigator's dead
# reckoning isn't calibrated - see guidance/guided_drive.py).
# MAX_LAUNCH_DURATION_S is a safety cap in case contact is never detected
# (ball slips away, detection glitch) - without it a missed contact would
# drive forward indefinitely (the wall-safety check still overrides
# regardless, but this is a tighter, ball-specific bound).
#
# Maxed out (60 -> 95, not all the way to 100 to leave a hair of margin) -
# unlike strafe, forward/backward drive both channels on each board in the
# SAME direction, so they don't have the same-board-opposite-channel
# coupling that limits strafe's safe ceiling. Ramming as hard as possible
# is the whole point of this maneuver, and the soft-start ramp in
# movement.py already tapers the inrush regardless of target speed.
LAUNCH_SPEED = 95
CONTACT_Y_THRESHOLD = int(FRAME_HEIGHT * 0.92)  # PLACEHOLDER - tune against where the ball visually disappears/touches at contact
# The ball's mask covering at least this fraction of the frame is a much
# more reliable "actually close" signal than Y position alone - a ball
# that's still 50cm-1m away can fall out of the ROI/frame entirely just
# from the camera's downward FOV not reaching that far along the floor,
# which used to get misread as "touching" (it vanished -> must have
# contacted) and cut the ram short after just ~10cm. Area only balloons
# once it's genuinely close, regardless of where in frame it sits.
#
# Raised well above the old 0.12 - that was tuned when LAUNCH_SPEED was
# 60; now that it's maxed to 95, the ram covers so much ground within
# just the MIN_LAUNCH_DURATION_S floor that 0.12 was getting crossed
# almost instantly, declaring "contact" after barely any real distance,
# then symmetrically returning for that same tiny duration, then
# immediately re-triggering - a rapid forward/backward oscillation
# instead of one real ram. This forces it to actually get close before
# calling it done, regardless of ram speed. PLACEHOLDER - tune against
# the ball's actual apparent size right at contact.
CONTACT_AREA_FRACTION = 0.35
MAX_LAUNCH_DURATION_S = 3.5

# CLOSE_Y_THRESHOLD (where the launch triggers) and CONTACT_Y_THRESHOLD
# (where it's declared done) aren't far apart - if the ball's already
# near CONTACT_Y_THRESHOLD the instant the launch starts, or detection is
# just noisy for a frame, contact could register on literally the very
# next step. Without a floor on how long the forward phase must run, that
# cuts it down to a near-zero-duration blip - too brief for the motor's
# soft-start ramp (movement.py) to produce any real movement, so what's
# actually visible ends up being just the backward return. This
# guarantees a real, visible push every time regardless of how fast
# contact gets detected.
MIN_LAUNCH_DURATION_S = 0.3

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
# search moves by CORD (Navigator's tracked X, drive.pose()[0]) instead of
# an angle: strafe left until X has moved SEARCH_STRAFE_CM, checking for
# the ball after every short burst, then (if still nothing) back to center
# and the same distance right. Heading never changes at all here, so
# unlike an angle-based sweep there's no "return to home heading"
# correction needed for that - but it still returns toward the starting X
# if BOTH sides come up empty, so a failed search doesn't leave it
# drifted away from its defensive spot for no reason. PLACEHOLDER - tune
# distance/speed against the real camera's field of view and how well
# dead reckoning tracks over that short a move.
SEARCH_STRAFE_CM = 30
SEARCH_STRAFE_TIMEOUT_S = 3.0  # safety cap in case dead-reckoning drift means the target X is never quite reached
SEARCH_PAUSE_S = 0.4

# Mecanum strafing doesn't always track perfectly straight - uneven wheel
# speeds/trim mean it can visibly curve/rotate instead of moving purely
# sideways, which compounds into a noticeable "turn" over a multi-burst
# search. _search_for_ball_straight() is an alternate to _search_for_ball()
# that corrects heading back to HOME_HEADING_DEG after every burst if it's
# drifted past STRAIGHT_HEADING_TOLERANCE_DEG. Costs a bit of extra time
# per burst (an occasional rotate_to() call) for straighter tracking.
HOME_HEADING_DEG = 0.0
STRAIGHT_HEADING_TOLERANCE_DEG = 5

# Original rotate-based search (before the cord/strafe search became the
# default) - kept as a selectable option, not deleted: SEARCH_SWEEP_DEG
# (45) each way, one big rotate_to() jump per side with a pause-and-check,
# always returning to HOME_HEADING_DEG afterward.
SEARCH_SWEEP_DEG = 45

# Newer rotate-based search: sweeps continuously (small steps, checking
# every step - no discrete jump-then-pause) up to ROTATE_SEARCH_SWEEP_DEG
# (90) each direction instead of a fixed 45, stopping the instant the ball
# shows up rather than only checking at two fixed points. Runs at
# ROTATE_SEARCH_SPEED (faster than the cautious 30 used elsewhere for
# search rotation) since continuous per-step checking means it doesn't
# need to move slowly to avoid blurring past the ball between checks the
# way a big blind jump would - some motion-blur risk traded for reaction
# speed. Always returns to HOME_HEADING_DEG afterward, same as the
# original sweep.
ROTATE_SEARCH_SWEEP_DEG = 90
ROTATE_SEARCH_SPEED = 45
ROTATE_SEARCH_STEP_DELAY_S = 0.05

# Which _search_for_ball_* variant defend_step uses when the ball isn't
# found - "cord" | "cord_straight" | "rotate_sweep" | "rotate_continuous".
# All four are kept side by side (see the functions below) so this is a
# one-line switch to compare them instead of losing old ones as new ones
# get added.
SEARCH_MODE = "cord"

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


# The robot chassis is a distinctive medium/royal blue 3D-printed frame -
# if this shows up, treat it as a target too, same as the ball (center on
# it, ram it once close). Estimated from photos only, NOT sampled
# on-site - PLACEHOLDER, verify/tune against cam_control.py's
# segmentation view before trusting this on the real chassis under real
# lighting.
CHASSIS_BLUE_LOW = np.array([100, 100, 50])
CHASSIS_BLUE_HIGH = np.array([130, 255, 255])


def _blue_mask(frame):
    # Own small HSV check, independent of perception.get_masks() (which
    # only knows ball/wall/floor). Same floor-only ROI restriction as the
    # ball, to keep background out.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CHASSIS_BLUE_LOW, CHASSIS_BLUE_HIGH)
    mask[:BALL_ROI_Y_START, :] = 0
    return mask


def _largest_centroid(mask, min_area):
    # Largest-contour-by-area centroid - no circularity filter, since the
    # chassis is boxy, not round (unlike the ball, this doesn't need
    # color_segment.ball_center()'s exact behavior, just its idea).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


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


def _find_target(frame, masks):
    # Ball first, blue chassis (opponent) as fallback - same priority as
    # the main per-frame check in defend_step, applied consistently across
    # every search variant so the search hunts for either one, not just
    # the ball.
    center = ball_center(_ball_roi(masks["ball"]))
    if center is not None:
        return center
    return _largest_centroid(_blue_mask(frame), min_area=30)


def _strafe_until(cap, drive, target_x, move):
    # Blocking: strafes in short bursts (move = "strafe_left" or
    # "strafe_right"), checking the tracked X (Navigator's dead reckoning
    # - cord, not an angle/heading) and the target after every burst.
    # Stops and returns the instant either it's spotted or target_x is
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
            center = _find_target(frame, masks)
            if center is not None:
                return frame, masks, center
    drive.stop()
    return None, None, None


def _search_for_ball(cap, drive):
    # Ball's not visible - strafe toward +-SEARCH_STRAFE_CM (tracked X,
    # cord not angle) checking for it after every short burst. Stops as
    # soon as it's spotted and just continues from wherever it currently
    # is (already looking right at it) rather than forcing a return to
    # the starting X first - only returns there if BOTH sides come up
    # empty, so a failed search doesn't leave the bot drifted off its
    # defensive spot for no reason.
    start_x = drive.pose()[0]

    frame, masks, center = _strafe_until(cap, drive, start_x - SEARCH_STRAFE_CM, "strafe_left")
    if center is not None:
        return frame, masks, center

    # Back to center before trying the other side, so each leg covers the
    # same ~SEARCH_STRAFE_CM distance instead of compounding.
    _strafe_until(cap, drive, start_x, "strafe_right")

    frame, masks, center = _strafe_until(cap, drive, start_x + SEARCH_STRAFE_CM, "strafe_right")
    if center is not None:
        return frame, masks, center

    current_x = drive.pose()[0]
    _strafe_until(cap, drive, start_x, "strafe_left" if current_x > start_x else "strafe_right")
    return None, None, None


def _strafe_straight_until(cap, drive, target_x, move):
    # Same idea as _strafe_until(), plus a heading check/correction after
    # every burst - mecanum strafing can drift off a straight line (uneven
    # wheel speeds/trim), and left uncorrected that compounds into a
    # visible curve/turn over several bursts. SEARCH_MODE = "cord_straight".
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
            center = _find_target(frame, masks)
            if center is not None:
                return frame, masks, center
    drive.stop()
    return None, None, None


def _search_for_ball_straight(cap, drive):
    # Alternate to _search_for_ball() - same cord-based (not angle) idea,
    # but actively corrects heading drift after each burst so the strafe
    # actually tracks straight sideways instead of curving over the
    # course of the search. SEARCH_MODE = "cord_straight" to use this one.
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


def _search_for_ball_rotate_sweep(cap, drive):
    # Original rotate-based search (the very first version of this
    # function, before the cord-based strafe search became the default) -
    # kept as a selectable option, not deleted. Blind rotate_to() calls
    # block until each turn completes (or times out): look left, pause and
    # check, look right, pause and check. Returns as soon as either side
    # spots something. Always rotates back to HOME_HEADING_DEG before
    # returning - whether or not anything was found - then takes one more
    # fresh frame there so the L/C/R zone read afterward is relative to
    # the home-facing view, not whatever angle it was found at.
    # SEARCH_MODE = "rotate_sweep" to use this one.
    found = False
    for offset in (-SEARCH_SWEEP_DEG, SEARCH_SWEEP_DEG):
        drive.rotate_to((HOME_HEADING_DEG + offset) % 360)
        time.sleep(SEARCH_PAUSE_S)
        ret, frame = cap.read()
        if ret and _find_target(frame, get_masks(frame)) is not None:
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
    return frame, masks, _find_target(frame, masks)


def _rotate_sweep_side(cap, drive, move):
    # Continuously rotates (small steps, checking every step) up to
    # ROTATE_SEARCH_SWEEP_DEG in one direction, stopping the instant the
    # ball's spotted. Unlike _search_for_ball_rotate_sweep's one big blind
    # rotate_to() jump per side, this checks constantly along the way, so
    # it can afford to move faster without as much risk of blurring past
    # the ball between checks.
    swept = 0.0
    last_heading = drive.pose()[2]
    while swept < ROTATE_SEARCH_SWEEP_DEG:
        getattr(drive, move)(speed=ROTATE_SEARCH_SPEED)
        time.sleep(ROTATE_SEARCH_STEP_DELAY_S)
        heading = drive.pose()[2]
        swept += abs((heading - last_heading + 180) % 360 - 180)
        last_heading = heading

        ret, frame = cap.read()
        if ret and _find_target(frame, get_masks(frame)) is not None:
            return True
    return False


def _search_for_ball_rotate_continuous(cap, drive):
    # Newer rotate-based search - see the ROTATE_SEARCH_* comment above.
    # SEARCH_MODE = "rotate_continuous" to use this one.
    home_heading = drive.pose()[2]
    found = False
    for move in ("rotate_left", "rotate_right"):
        if _rotate_sweep_side(cap, drive, move):
            found = True
            break
        drive.rotate_to(home_heading)  # this side exhausted - reset before trying the other

    drive.stop()
    drive.rotate_to(home_heading)
    if not found:
        return None, None, None

    time.sleep(ROTATE_SEARCH_STEP_DELAY_S)
    ret, frame = cap.read()
    if not ret:
        return None, None, None
    masks = get_masks(frame)
    return frame, masks, _find_target(frame, masks)


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


def defend_step(cap, drive, search_state=None):
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    ball_mask = _ball_roi(masks["ball"])
    center = ball_center(ball_mask)
    target_mask = ball_mask
    if center is None:
        # No ball this frame - check for the opponent chassis (blue) as a
        # fallback target. Same zone/launch machinery below then centers
        # on and rams whichever one was actually found - target_mask
        # tracks which one, so the ram's contact-area check (below) reads
        # area from the right mask instead of always assuming the ball.
        blue_mask = _blue_mask(frame)
        blue_center = _largest_centroid(blue_mask, min_area=30)
        if blue_center is not None:
            center = blue_center
            target_mask = blue_mask

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
            # Keep ramming forward until actual contact - the ball's mask
            # covering CONTACT_AREA_FRACTION of the frame (genuinely
            # close, not just "near the bottom of frame") - instead of a
            # fixed time burst that might stop short of really reaching
            # it. If the ball vanishes from view without ever having read
            # that large, that's the camera's FOV running out at range,
            # not contact - don't count it (keep ramming blind toward
            # where it was last seen, up to MAX_LAUNCH_DURATION_S, which
            # is a safety cap in case contact is never detected at all;
            # the wall-safety check above still overrides every step
            # regardless). If it WAS already large just before vanishing,
            # that's a real hit landing right as the ball fills/exceeds
            # the frame - count that as contact.
            ball_area = int((target_mask > 0).sum())
            frame_area = frame.shape[0] * frame.shape[1]
            large_enough = ball_area >= CONTACT_AREA_FRACTION * frame_area
            if center is not None and large_enough:
                search_state["launch_max_area_seen"] = True
            contact = (center is not None and large_enough) or (
                center is None and search_state.get("launch_max_area_seen", False)
            )
            # Contact can only end the push early once MIN_LAUNCH_DURATION_S
            # has actually elapsed - guarantees a real push every time
            # instead of contact registering before the wheels have even
            # ramped up.
            if elapsed < MIN_LAUNCH_DURATION_S or (not contact and elapsed < MAX_LAUNCH_DURATION_S):
                _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
                return f"Ramming forward ({elapsed:.1f}s)"
            search_state["launch_phase"] = "returning"
            search_state["launch_return_duration"] = elapsed
            search_state["launch_start"] = time.time()
            elapsed = 0.0
        return_duration = search_state.get("launch_return_duration", MAX_LAUNCH_DURATION_S)
        if elapsed < return_duration:
            if _safe_move(drive, search_state, "backward", LAUNCH_SPEED):
                return f"Returning to position ({elapsed:.1f}s)"
            return f"Braking before returning ({elapsed:.1f}s)"
        search_state.pop("launch_phase", None)
        search_state.pop("launch_start", None)
        search_state.pop("launch_return_duration", None)
        search_state.pop("launch_max_area_seen", None)
        return "Launch complete - resuming defense"

    if center is None:
        frame, masks, center = _SEARCH_MODES[SEARCH_MODE](cap, drive)
        search_state["last_move"] = "stop"  # every _search_for_ball_* variant always ends stopped
        if center is None:
            _stop(drive, search_state)
            return "No ball found (strafed L/R) - holding position"

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
