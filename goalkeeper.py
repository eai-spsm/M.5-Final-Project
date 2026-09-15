import time

import cv2
import numpy as np

from guidance import GuidedDrive
from perception import get_masks, ball_center, find_goal_gap

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
# Was pushed 60 -> 65 -> 75 -> 85 in one earlier session without ever
# confirming on real hardware whether the same-board-opposite-channel
# glitch (a wheel visibly flipping direction mid-hold, see
# CALIBRATION_REPORT.md) or a brownout (movement/movement.py's soft-start
# exists because current spikes were already dropping the Pi at lower
# speeds) showed up at any of those steps. Reset to the documented safe
# ceiling (60), then stepped back up in smaller +5 increments (65 -> 70) -
# stay alert for the direction-flip glitch; if it shows up, that's
# confirmation to drop back down rather than stepping further up.
ZONE_LEFT_FRACTION = 1 / 3
ZONE_RIGHT_FRACTION = 2 / 3
STRAFE_SPEED = 70

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

# The backward return after a ram used to be fully time-symmetric (return
# duration == however long the forward push took) to land back at exactly
# the pre-ram spot. Shortened here so it doesn't spend as long backing up
# as it did ramming forward - trades some drift forward over repeated
# rams (it won't return to the *exact* pre-ram spot) for spending less
# time out of position/facing the wrong way after every hit. PLACEHOLDER -
# if it drifts too far forward from its post over a match, raise this back
# toward 1.0.
RETURN_DURATION_FRACTION = 0.6

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

# If the ball genuinely vanishes mid-ram (missed it, it deflected away) and
# was never seen large enough to count as contact, don't keep ramming blind
# all the way to MAX_LAUNCH_DURATION_S - abort the forward push once it's
# been gone for this long and drop straight into the normal time-symmetric
# "returning" phase (using whatever forward duration had already elapsed),
# which puts the robot back at roughly its pre-ram position instead of
# overshooting into empty space. Not zero - a single dropped/flickered
# frame shouldn't abort a real ram, same reasoning as the near-field-motion
# and lost-track grace periods elsewhere.
LOST_ABORT_GRACE_S = 0.4

# Once centered (heading locked onto the ball) but not yet close enough to
# trigger the ram, approach at a speed that ramps up with how close the
# ball already is (its Y position toward CLOSE_Y_THRESHOLD) instead of
# just holding still and waiting - a ball that's still approaching from
# far off gets a gentle nudge forward, and speed climbs toward LAUNCH_SPEED
# as it nears the ram trigger, rather than sitting still then jumping
# straight to max speed the instant CLOSE_Y_THRESHOLD is crossed.
APPROACH_MIN_SPEED = 45

# If the wall mask covers this much of the frame, back up regardless of
# what the ball's doing or mid-maneuver - basic collision safety, same as
# main.py.
WALL_COVERAGE_THRESHOLD = 0.80

# find_goal_gap() reads the goal's crossbar/frame opening out of the wall
# mask - a wide detected gap means we're close enough to be looking right
# into (or through) the goal structure itself, not just a flat wall. Same
# check and threshold as main.py's "don't go into the goal" - a goalkeeper
# ramming/drifting itself into its own goal is exactly as bad as an
# attacker chasing the ball into one. Highest-priority failsafe: overrides
# everything else, including an in-progress ram or pin recovery.
GOAL_GAP_CLOSE_WIDTH_FRACTION = 0.5

# Backstop for the goal-gap check above: find_goal_gap() is built to
# recognize looking at the goal's crossbar/frame opening from a distance -
# up close, with the ball centered right against the goal's sill/threshold,
# that pattern isn't there anymore and the check can simply never fire.
# This doesn't look at the camera at all - it measures how far forward
# (dead-reckoned Y, drive.pose()) the current unbroken forward push has
# covered - approach creep and the final ram both count as one continuous
# excursion, since neither ever sets last_move to anything but "forward"
# in between - and refuses to keep pushing forward past that regardless of
# what vision says. Independent of any single perception blind spot, same
# "vision OR position, whichever fires first" idea as the goal-gap check.
#
# Must clear whatever a normal, intended approach+ram actually needs to
# travel to reach and hit the ball - set too low (40 was too tight; cut
# real rams short before contact, causing "det C" / "ramming" / "too far"
# to flicker rapidly instead of ever finishing a hit) and this fires on
# every single legitimate ram instead of only the genuinely-too-far case.
# PLACEHOLDER - tune against how far it's actually safe to advance from a
# real defensive post before reaching the goal line; raise further if
# normal rams still get cut short.
MAX_FORWARD_DRIFT_CM = 100.0

# WALL_COVERAGE_THRESHOLD assumes ramming into something tall enough to
# fill most of the frame - it misses being jammed against a low physical
# barrier (a rail, a curb, a bumper-height obstacle) that only ever fills
# the near-field band at the bottom of view while the rest of the frame
# still shows whatever's behind/above it. Checked against just that
# bottom band (NEAR_ROI_Y_START) instead of the whole frame, so this can
# be a much lower fraction than WALL_COVERAGE_THRESHOLD and still mean
# "pinned" - PLACEHOLDER, tune against real snapshots of being stuck on
# something low.
NEAR_WALL_COVERAGE_THRESHOLD = 0.50

# Backup for anything the wall-color checks above miss entirely (wrong
# color, glass, another robot) - doesn't look at color at all, just
# whether the view is actually changing while forward motion is being
# commanded. If it stays essentially static for STUCK_CONFIRM_S straight
# while "last_move" is forward, something's physically blocking it
# regardless of what it looks like.
STUCK_MOTION_DIFF_THRESHOLD = 15
STUCK_MOTION_PIXEL_FRACTION = 0.05
STUCK_CONFIRM_S = 0.5

# How many consecutive near-identical ram frames (_ram_stalled) before
# declaring the ram physically wedged - much shorter than STUCK_CONFIRM_S
# since contact-area detection can otherwise resolve the ram faster than
# a longer confirm window would ever get a chance to fire. 1 would react
# on a single noisy/flickered frame; 2 needs it to actually hold still.
RAM_STALL_STREAK = 2

# Once pinned, just reversing the wheels doesn't always work - if it's
# genuinely wedged (a low rail catching the chassis at an angle, not just
# resting against a flat wall), driving straight backward can push against
# the same jam instead of clearing it. Turning 180 degrees in place first
# re-aims the front where the back was and vice versa, then driving
# "forward" along that new facing is a different escape vector than
# whatever direction caused the pin - more likely to actually break free.
# Finishes by rotating back to HOME_HEADING_DEG so defense resumes facing
# the right way, not backward. PLACEHOLDER - tune PIN_AWAY_DURATION_S/
# PIN_AWAY_SPEED against how long it actually takes to clear something.
PIN_AWAY_SPEED = 60
PIN_AWAY_DURATION_S = 0.5

# If the ball's actually visible in frame at the same moment we're pinned,
# a full 180 flip-away abandons it and backs off further than needed. Try
# a "hook" instead, alternating sides up to PIN_WALL_PUSH_ATTEMPTS times:
# back away, offset sideways, then ram forward from that new angle - not
# a straight-sideways strafe against the ball. Approaching from an angled
# offset gives more actual leverage against something wedged than nudging
# it directly sideways in place does (same reasoning as PIN_AWAY's 180
# flip - a different approach vector, more likely to break something
# genuinely stuck free), while still working the ball toward one side
# instead of abandoning position with a full flip-away. Only falls
# through to the full 180 flip-away if none of the attempts actually
# clear the pin. PLACEHOLDER - tune durations/speed/attempt count against
# how it actually behaves against a real wall+ball corner.
PIN_HOOK_BACK_S = 0.25
PIN_HOOK_OFFSET_S = 0.25
PIN_HOOK_RAM_S = 0.35
PIN_WALL_PUSH_SPEED = 55
PIN_WALL_PUSH_ATTEMPTS = 3

# The ultrasonic sensor is mounted at the REAR - it watches for something
# (opponent, ball) closing in from behind while attention's on the front
# camera. Can't tell which side it's coming from, and strafing wouldn't
# increase distance from a rear threat anyway - push forward instead,
# same reasoning and same values as main.py's rear-threat evade.
REAR_THREAT_DISTANCE_CM = 15
EVADE_SPEED = 60

# If the camera read fails this many times in a row, assume it's actually
# disconnected and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0

# The whole point of this strategy is staying facing outward and using
# strafes (not turns) to track/move - so when the ball isn't visible, the
# search moves by CORD (Navigator's tracked X, drive.pose()[0]) instead of
# an angle: strafe left until X has moved SEARCH_STRAFE_CM, checking for
# the ball continuously along the way, then (if still nothing) back to
# center and the same distance right. Heading never changes at all here,
# so unlike an angle-based sweep there's no "return to home heading"
# correction needed for that - but it still returns toward the starting X
# if BOTH sides come up empty, so a failed search doesn't leave it
# drifted away from its defensive spot for no reason.
#
# Widened from 30 - now that _strafe_until() checks continuously instead
# of stop-and-check bursts, covering more ground doesn't cost nearly as
# much extra time as it used to, so there's less reason to keep the sweep
# this narrow. PLACEHOLDER - tune distance/speed against the real camera's
# field of view and how well dead reckoning tracks over that longer a
# move (more distance covered = more room for open-loop drift to
# accumulate before target_x is actually reached).
SEARCH_STRAFE_CM = 45
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
# search rotation) since per-step checking means it doesn't need to move
# slowly to avoid blurring past the ball between checks the way a big
# blind jump would. Always returns to HOME_HEADING_DEG afterward, same as
# the original sweep.
#
# _rotate_sweep_side used to capture WHILE STILL ACTIVELY ROTATING (never
# stopped between the move command and the check) - real angular step per
# check was small (~3-4deg at this speed/delay), so it wasn't skipping
# past the ball's angular window, but every single check frame was
# motion-blurred from spinning at full ROTATE_SEARCH_SPEED the instant it
# was captured, which is what was actually causing "spins too fast to
# find it". ROTATE_SEARCH_SETTLE_S adds a full stop + brief settle before
# each check (stop-and-look, same fix as the ram/rotate search elsewhere)
# so every check frame is sharp regardless of rotation speed.
ROTATE_SEARCH_SWEEP_DEG = 90
# Lowered from 45 - stop-and-look already fixed the pure motion-blur
# problem, but a slower rotation still gives the settle step a gentler
# starting point to actually stop from (less momentum to kill) and a
# smaller real angular step per check, both of which make it less likely
# to swing past the ball between look-points.
ROTATE_SEARCH_SPEED = 30
ROTATE_SEARCH_STEP_DELAY_S = 0.05
ROTATE_SEARCH_SETTLE_S = 0.15

# Which _search_for_ball_* variant defend_step uses when the ball isn't
# found - "cord" | "cord_straight" | "rotate_sweep" | "rotate_continuous" |
# "combined". All variants are kept side by side (see the functions below)
# so this is a one-line switch to compare them instead of losing old ones
# as new ones get added. "combined" (rotate sweep first, cord fallback) is
# now the default - wider angular coverage than pure strafing, without
# losing the strafe search entirely.
SEARCH_MODE = "combined"

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


def _heading_diff(drive):
    # Signed degrees off HOME_HEADING_DEG, wrapped to -180..180. Used to
    # make sure the robot is actually squared up before ramming forward -
    # search bursts already self-correct back toward home heading, but
    # nothing previously checked this right before a launch, so a launch
    # fired while still rotated off-heading could deflect the ball
    # sideways or back toward our own goal instead of straight out.
    heading = drive.pose()[2]
    return (heading - HOME_HEADING_DEG + 180) % 360 - 180


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


def _near_wall_pinned(masks):
    # Same idea as the whole-frame WALL_COVERAGE_THRESHOLD check, but
    # restricted to the near-field band at the bottom of the frame - catches
    # being jammed against something low that never fills most of the view.
    near_wall = masks["wall"][NEAR_ROI_Y_START:, :]
    return float((near_wall > 0).mean()) >= NEAR_WALL_COVERAGE_THRESHOLD


def _wall_flanks_target(masks):
    # _near_wall_pinned checks the WHOLE near-band width - but exactly
    # when a ball is centered right up against a wall (the case that
    # matters most for the ram-contact wall-push check), the ball's own
    # green blob occludes the center of that band, pulling the average
    # coverage down and making the check fail right when it should fire.
    # This checks only the LEFT and RIGHT thirds instead, where the ball
    # (usually centered) isn't blocking the view - a wall/rail visible on
    # both flanks at close range is a strong signal regardless of what's
    # occluding the middle.
    near_wall = masks["wall"][NEAR_ROI_Y_START:, :]
    width = near_wall.shape[1]
    third = width // 3
    left = near_wall[:, :third]
    right = near_wall[:, -third:]
    left_frac = float((left > 0).mean())
    right_frac = float((right > 0).mean())
    return left_frac >= NEAR_WALL_COVERAGE_THRESHOLD and right_frac >= NEAR_WALL_COVERAGE_THRESHOLD


def _ram_stalled(frame, search_state):
    # Wall-color checks (_near_wall_pinned, _wall_flanks_target) can still
    # miss a wedge if the ball's grown big enough to cover the flanks too,
    # or if what it's wedged against isn't the tuned wall color at all.
    # This doesn't look at color - just whether the view is changing at
    # all frame-to-frame while ramming at full speed. A couple of
    # consecutive near-identical frames (RAM_STALL_STREAK) means nothing's
    # actually still traveling, regardless of what's blocking it or
    # whether the ball's mask has crossed CONTACT_AREA_FRACTION yet - a
    # ram wedged at an angle might never grow to fill that much of the
    # frame at all. Deliberately a much shorter confirm window than
    # _is_stuck's STUCK_CONFIRM_S: that one was tuned conservatively for
    # the general case, but here contact-area detection can resolve (and
    # move the ram on to "returning") faster than a 0.5s window would ever
    # get to fire, letting a real wedge slip through undetected.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev = search_state.get("ram_prev_gray")
    search_state["ram_prev_gray"] = gray
    if prev is None or prev.shape != gray.shape:
        search_state["ram_static_streak"] = 0
        return False
    diff = cv2.absdiff(gray, prev)
    changed_fraction = float((diff > STUCK_MOTION_DIFF_THRESHOLD).mean())
    if changed_fraction >= STUCK_MOTION_PIXEL_FRACTION:
        search_state["ram_static_streak"] = 0
        return False
    search_state["ram_static_streak"] = search_state.get("ram_static_streak", 0) + 1
    return search_state["ram_static_streak"] >= RAM_STALL_STREAK


def _is_stuck(frame, search_state):
    # Color-independent backup: is the view actually changing while we're
    # commanding forward motion? If it stays essentially static for
    # STUCK_CONFIRM_S while last_move is "forward", something's physically
    # blocking movement regardless of what it looks like (wrong-colored
    # obstacle, glass, another robot). Only meaningful while genuinely
    # trying to drive forward - gated the same way as _near_field_motion,
    # just on the opposite last_move value.
    if search_state.get("last_move") != "forward":
        search_state.pop("stuck_ref", None)
        search_state.pop("stuck_since", None)
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref = search_state.get("stuck_ref")
    search_state["stuck_ref"] = gray
    now = time.time()
    if ref is None or ref.shape != gray.shape:
        search_state["stuck_since"] = now
        return False

    diff = cv2.absdiff(gray, ref)
    changed_fraction = float((diff > STUCK_MOTION_DIFF_THRESHOLD).mean())
    if changed_fraction >= STUCK_MOTION_PIXEL_FRACTION:
        search_state["stuck_since"] = now
        return False

    return (now - search_state.get("stuck_since", now)) >= STUCK_CONFIRM_S


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


# Strafing purely toward the ball's CURRENT x-position always reacts a beat
# late - by the time it's centered, a fast diagonal shot has already moved
# on. Instead, estimate lateral velocity from frame-to-frame x movement and
# aim at where the ball will be after LEAD_TIME_S, not where it is now.
# LEAD_TIME_S is a rough stand-in for total reaction+strafe-spin-up delay,
# not a measured value - PLACEHOLDER, tune against how early it actually
# needs to start moving to catch a fast real shot.
LEAD_TIME_S = 0.2
# Cap how far the prediction can shift the aim point - raw per-frame
# velocity from a small, noisy 320x240 centroid is jumpy, and an
# uncapped extrapolation on a bad frame could aim it wildly off to one
# side. This bounds the worst case to a fraction of the frame width.
MAX_LEAD_PX = int(FRAME_WIDTH * 0.35)
# If the gap since the last tracked frame is bigger than this, the ball
# was probably just reacquired after a search/loss (not continuously
# tracked) - a velocity computed across that gap would be meaningless, so
# skip the lead and use the raw position for one frame instead.
MAX_VELOCITY_DT_S = 0.5


def _lead_adjusted_x(x, kind, search_state):
    # kind distinguishes ball vs. blue-chassis targets - switching between
    # them mid-track is a jump in what's being measured, not real motion,
    # so that case also skips the lead for one frame (same as a stale gap).
    now = time.time()
    prev_x = search_state.get("track_prev_x")
    prev_t = search_state.get("track_prev_t")
    prev_kind = search_state.get("track_prev_kind")
    search_state["track_prev_x"] = x
    search_state["track_prev_t"] = now
    search_state["track_prev_kind"] = kind
    if prev_x is None or prev_kind != kind:
        return x
    dt = now - prev_t
    if dt <= 0 or dt > MAX_VELOCITY_DT_S:
        return x
    lead_px = max(-MAX_LEAD_PX, min(MAX_LEAD_PX, ((x - prev_x) / dt) * LEAD_TIME_S))
    return x + lead_px


def _find_target(frame, masks):
    # Ball only for now - blue-chassis (opponent) ramming is disabled, not
    # deleted, in case it's wanted again later. Was: ball first, blue
    # chassis as fallback, applied consistently across every search
    # variant so the search hunted for either one, not just the ball.
    return ball_center(_ball_roi(masks["ball"]))
    # center = ball_center(_ball_roi(masks["ball"]))
    # if center is not None:
    #     return center
    # return _largest_centroid(_blue_mask(frame), min_area=30)


def _strafe_until(cap, drive, target_x, move):
    # Blocking: strafes CONTINUOUSLY (no stop-and-check burst pattern),
    # checking every loop iteration against the tracked X (Navigator's
    # dead reckoning - cord, not an angle/heading) and the target. Used to
    # stop and settle before every single check (SEARCH_PAUSE_S), but that
    # stop/restart cycle plus the fixed pause cost real time on every
    # iteration regardless of how much ground actually needed covering -
    # removed since the ball is a large, easy color blob with no
    # circularity filter here, unlike the rotate search's motion-blur
    # problem, so it doesn't need a fully-stopped, blur-free frame to spot
    # it. Stops and returns the instant either it's spotted or target_x is
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
        ret, frame = cap.read()
        if ret:
            masks = get_masks(frame)
            center = _find_target(frame, masks)
            if center is not None:
                drive.stop()
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
    # Small steps, stop-and-look at every step, up to ROTATE_SEARCH_SWEEP_DEG
    # in one direction - stopping the instant the ball's spotted. Unlike
    # _search_for_ball_rotate_sweep's one big blind rotate_to() jump per
    # side, this checks constantly along the way, so it can afford to move
    # faster without skipping past the ball's angular window between
    # checks. Used to capture WHILE STILL ROTATING (no stop between the
    # move command and the check) - that made every check frame
    # motion-blurred regardless of how small the per-step angle was, which
    # is what was actually causing it to spin right past the ball without
    # recognizing it. Now comes to a full stop and settles briefly first,
    # so every check frame is sharp.
    swept = 0.0
    last_heading = drive.pose()[2]
    while swept < ROTATE_SEARCH_SWEEP_DEG:
        getattr(drive, move)(speed=ROTATE_SEARCH_SPEED)
        time.sleep(ROTATE_SEARCH_STEP_DELAY_S)
        drive.stop()
        time.sleep(ROTATE_SEARCH_SETTLE_S)
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


def _search_for_ball_combined(cap, drive):
    # Strafe search is the main/first search - runs a full cycle (left,
    # back to center, right, back to center - see _search_for_ball) since
    # that's the more dead-reckoning-repeatable search and keeps the bot
    # closer to its actual defensive line the whole time. Only if BOTH
    # strafe legs come up empty does it fall back to the wider-angle
    # rotate search, which covers angles pure strafing structurally can't
    # (it never turns the camera, only slides sideways while still facing
    # forward) - a last resort, not the first try, since it takes the bot
    # off its home heading for longer.
    #
    # Uses the CONTINUOUS rotate variant (_search_for_ball_rotate_continuous
    # / _rotate_sweep_side), not _search_for_ball_rotate_sweep - that one
    # only ever checks at two blind endpoints (-45deg, +45deg) with a big
    # blocking rotate_to() between them, so a ball anywhere in between
    # those two exact angles is swept straight past unseen. The continuous
    # variant checks in small steps throughout the whole sweep instead, so
    # it actually covers the angles in between, not just the endpoints.
    frame, masks, center = _search_for_ball(cap, drive)
    if center is not None:
        return frame, masks, center
    return _search_for_ball_rotate_continuous(cap, drive)


_SEARCH_MODES = {
    "cord": _search_for_ball,
    "cord_straight": _search_for_ball_straight,
    "rotate_sweep": _search_for_ball_rotate_sweep,
    "rotate_continuous": _search_for_ball_rotate_continuous,
    "combined": _search_for_ball_combined,
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
    # Blue-chassis (opponent) ramming fallback disabled, not deleted - in
    # case it's wanted again later. Was: fall back to the largest blue
    # blob as the target whenever the ball isn't seen, with target_mask
    # tracking which one so the ram's contact-area check below reads area
    # from the right mask instead of always assuming the ball.
    # if center is None:
    #     blue_mask = _blue_mask(frame)
    #     blue_center = _largest_centroid(blue_mask, min_area=30)
    #     if blue_center is not None:
    #         center = blue_center
    #         target_mask = blue_mask

    # --- Highest-priority failsafe: don't ram/drift into our own goal ---
    # Overrides everything below, including an in-progress ram or pin
    # recovery - being inside the goal opening is worse than whatever
    # those were handling. Two independent signals, either one triggers:
    # vision (find_goal_gap - can miss up-close/off-angle views where the
    # goal's crossbar pattern isn't recognizable) OR position (dead-
    # reckoned forward drift - doesn't depend on what the camera sees).
    #
    # The position reference resets to the current spot every time the
    # bot ISN'T actively driving forward (search, strafe, holding, back
    # up) and only accumulates across consecutive forward-only steps -
    # this measures "how far has THIS forward push gone", not cumulative
    # distance since the whole run started. RETURN_DURATION_FRACTION
    # deliberately doesn't return all the way after every ram, so
    # cumulative-since-start drift builds up over a match even nowhere
    # near the goal and would false-trigger this on a totally normal ram.
    if search_state.get("last_move") != "forward":
        search_state["forward_excursion_start_y"] = drive.pose()[1]
    forward_drift_cm = drive.pose()[1] - search_state.get(
        "forward_excursion_start_y", drive.pose()[1]
    )
    gap = find_goal_gap(masks["wall"])
    close_by_vision = gap is not None and gap["width_px"] >= GOAL_GAP_CLOSE_WIDTH_FRACTION * frame.shape[1]
    close_by_position = forward_drift_cm >= MAX_FORWARD_DRIFT_CM
    if close_by_vision or close_by_position:
        reason = "Near goal opening" if close_by_vision else f"Ramming {forward_drift_cm:.0f}cm straight, too far"
        search_state.pop("launch_phase", None)
        search_state.pop("pin_phase", None)
        search_state.pop("pin_push_start", None)
        search_state.pop("pin_push_direction", None)
        search_state.pop("pin_push_attempt", None)
        search_state.pop("pin_target_heading", None)
        search_state.pop("pin_away_start", None)
        if _safe_move(drive, search_state, "backward"):
            return f"{reason} - backing out"
        return f"{reason} - braking before backing out"

    wall_fraction = float((masks["wall"] > 0).mean())
    near_wall_pinned = _near_wall_pinned(masks)
    stuck = _is_stuck(frame, search_state)
    pinned = wall_fraction >= WALL_COVERAGE_THRESHOLD or near_wall_pinned or stuck
    if stuck:
        pin_reason = "Stuck (view static while driving forward)"
    elif near_wall_pinned:
        pin_reason = "Low obstacle fills near-field view"
    else:
        pin_reason = f"Wall fills {wall_fraction * 100:.0f}% of view"

    # Continue an in-progress pin-recovery maneuver before anything below
    # gets a chance to re-detect the same pin and restart it every frame.
    pin_phase = search_state.get("pin_phase")
    if pin_phase is not None:
        if pin_phase == "wall_push":
            if not pinned:
                search_state.pop("pin_phase", None)
                search_state.pop("pin_push_start", None)
                search_state.pop("pin_push_direction", None)
                search_state.pop("pin_push_attempt", None)
                search_state.pop("hook_subphase", None)
                _stop(drive, search_state)
                return "Unpinned - resuming defense"
            # "Hook" maneuver, not a straight-sideways strafe: back away,
            # offset to the OPPOSITE side of the target push direction,
            # then ram forward - approaching from that angled offset gives
            # more real leverage against something wedged than nudging it
            # directly sideways in place (same idea as PIN_AWAY's 180 flip
            # - a different approach vector). direction is which way we're
            # trying to push the ball; offset_move is the side we swing
            # out to first so the ram comes in from an angle.
            direction = search_state["pin_push_direction"]
            offset_move = "strafe_right" if direction == "strafe_left" else "strafe_left"
            subphase = search_state.get("hook_subphase", "back")
            sub_elapsed = time.time() - search_state["pin_push_start"]
            if subphase == "back":
                if sub_elapsed < PIN_HOOK_BACK_S:
                    if _safe_move(drive, search_state, "backward", PIN_WALL_PUSH_SPEED):
                        return f"Pinned with ball - hooking {direction}: backing off ({sub_elapsed:.1f}s)"
                    return f"Pinned with ball - braking before hooking {direction}"
                search_state["hook_subphase"] = "offset"
                search_state["pin_push_start"] = time.time()
                _stop(drive, search_state)
                return f"Pinned with ball - hooking {direction}: repositioning"
            if subphase == "offset":
                if sub_elapsed < PIN_HOOK_OFFSET_S:
                    if _safe_move(drive, search_state, offset_move, PIN_WALL_PUSH_SPEED):
                        return f"Pinned with ball - hooking {direction}: offsetting ({sub_elapsed:.1f}s)"
                    return f"Pinned with ball - braking before offsetting"
                search_state["hook_subphase"] = "ram"
                search_state["pin_push_start"] = time.time()
                _stop(drive, search_state)
                return f"Pinned with ball - hooking {direction}: ramming"
            # subphase == "ram"
            if sub_elapsed < PIN_HOOK_RAM_S:
                if _safe_move(drive, search_state, "forward", PIN_WALL_PUSH_SPEED):
                    return f"Pinned with ball - hooking {direction}: ram ({sub_elapsed:.1f}s)"
                return f"Pinned with ball - braking before ram"
            search_state.pop("hook_subphase", None)
            search_state["pin_push_attempt"] += 1
            if search_state["pin_push_attempt"] >= PIN_WALL_PUSH_ATTEMPTS:
                # Alternating hooks didn't clear it - fall back to the
                # full flip-away instead of giving up.
                search_state.pop("pin_push_start", None)
                search_state.pop("pin_push_direction", None)
                search_state.pop("pin_push_attempt", None)
                heading_now = drive.pose()[2]
                search_state["pin_phase"] = "turn"
                search_state["pin_target_heading"] = (heading_now + 180) % 360
                _stop(drive, search_state)
                return "Hook attempts exhausted - flipping 180 to reface"
            search_state["pin_push_direction"] = (
                "strafe_right" if direction == "strafe_left" else "strafe_left"
            )
            search_state["pin_push_start"] = time.time()
            _stop(drive, search_state)
            return f"Pinned with ball - trying other side (attempt {search_state['pin_push_attempt']})"
        if pin_phase == "turn":
            drive.rotate_to(search_state["pin_target_heading"])
            search_state["last_move"] = "stop"
            search_state["pin_phase"] = "away"
            search_state["pin_away_start"] = time.time()
            return "Pinned - flipped 180, clearing along new facing"
        if pin_phase == "away":
            elapsed = time.time() - search_state["pin_away_start"]
            if elapsed < PIN_AWAY_DURATION_S:
                if _safe_move(drive, search_state, "forward", PIN_AWAY_SPEED):
                    return f"Pinned - clearing obstacle ({elapsed:.1f}s)"
                return f"Pinned - braking before clearing ({elapsed:.1f}s)"
            search_state["pin_phase"] = "restore"
            return "Pinned - clear, restoring heading"
        # pin_phase == "restore"
        drive.rotate_to(HOME_HEADING_DEG)
        search_state["last_move"] = "stop"
        search_state.pop("pin_phase", None)
        search_state.pop("pin_target_heading", None)
        search_state.pop("pin_away_start", None)
        return "Unpinned - resuming defense"

    if pinned:
        search_state.pop("launch_phase", None)
        if center is not None:
            # Ball's right here too - try pushing it left, then right if
            # that doesn't clear it, alternating a few times instead of
            # immediately abandoning position with a full 180 flip-away.
            search_state["pin_phase"] = "wall_push"
            search_state["pin_push_direction"] = "strafe_left"
            search_state["pin_push_start"] = time.time()
            search_state["pin_push_attempt"] = 1
            _stop(drive, search_state)
            return f"{pin_reason} - ball in frame, pushing left"
        heading_now = drive.pose()[2]
        search_state["pin_phase"] = "turn"
        search_state["pin_target_heading"] = (heading_now + 180) % 360
        _stop(drive, search_state)
        return f"{pin_reason} - pinned, flipping 180 to reface"

    # --- Rear ultrasonic threat - overrides everything below, including an
    # in-progress ram, same as the wall-safety check above. ---------------
    distance = drive.get_distance()
    rear_threat = distance is not None and distance < REAR_THREAT_DISTANCE_CM
    if rear_threat:
        search_state.pop("launch_phase", None)
        if _safe_move(drive, search_state, "forward", EVADE_SPEED):
            return f"Rear threat at {distance:.0f}cm - pushing forward"
        return f"Rear threat at {distance:.0f}cm - braking before pushing forward"

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
            if center is not None:
                search_state["launch_last_seen"] = time.time()
            contact = (center is not None and large_enough) or (
                center is None and search_state.get("launch_max_area_seen", False)
            )
            # If the ball's been gone (not a single-frame flicker) for
            # LOST_ABORT_GRACE_S and was never seen large, it's genuinely
            # lost, not just out of the ROI at range while still tracked -
            # stop ramming blind toward nothing and return early instead of
            # continuing to MAX_LAUNCH_DURATION_S.
            genuinely_lost = not search_state.get(
                "launch_max_area_seen", False
            ) and (time.time() - search_state.get("launch_last_seen", search_state["launch_start"])) >= LOST_ABORT_GRACE_S
            # Checked every ram step, not just once contact triggers - a
            # ram wedged at an angle might never grow to cross
            # CONTACT_AREA_FRACTION at all, so waiting for "contact" before
            # ever checking for a wedge could miss it entirely. See
            # _ram_stalled for why this doesn't reuse the slower
            # STUCK_CONFIRM_S-based _is_stuck.
            stalled = elapsed >= MIN_LAUNCH_DURATION_S and _ram_stalled(frame, search_state)
            # Contact can only end the push early once MIN_LAUNCH_DURATION_S
            # has actually elapsed - guarantees a real push every time
            # instead of contact registering before the wheels have even
            # ramped up.
            if elapsed < MIN_LAUNCH_DURATION_S or (
                not contact and not stalled and not genuinely_lost and elapsed < MAX_LAUNCH_DURATION_S
            ):
                _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
                return f"Ramming forward ({elapsed:.1f}s)"
            search_state.pop("ram_prev_gray", None)
            search_state.pop("ram_static_streak", None)
            # "Contact"/stalled while the wall's also right there usually
            # means the ball got wedged against it, not cleanly hit - the
            # ram reads as a success (area/view-static triggered it) but
            # the ball never actually went anywhere because the wall
            # blocked it. Try alternating strafe nudges to walk it out of
            # the corner before just backing off and leaving it stuck
            # there. `stalled` alone is enough regardless of wall-color
            # signals - a static view while ramming at full speed means
            # something's physically blocking progress no matter what it
            # looks like. _wall_flanks_target checks the near-band's
            # LEFT/RIGHT thirds only, which the ball itself doesn't
            # occlude when centered - near_wall_pinned/wall_fraction alone
            # were unreliable here since the ball sitting right in front of
            # the wall blocks most of the very pixels those checks look at.
            ball_against_wall = (
                stalled
                or wall_fraction >= WALL_COVERAGE_THRESHOLD * 0.5
                or near_wall_pinned
                or _wall_flanks_target(masks)
            )
            if (contact or stalled) and ball_against_wall:
                search_state["launch_phase"] = "ram_wall_push"
                search_state["launch_return_duration"] = elapsed * RETURN_DURATION_FRACTION
                search_state["pin_push_direction"] = "strafe_left"
                search_state["pin_push_start"] = time.time()
                search_state["pin_push_attempt"] = 1
                return "Contact against wall - pushing left to free the ball"
            search_state["launch_phase"] = "returning"
            search_state["launch_return_duration"] = elapsed * RETURN_DURATION_FRACTION
            search_state["launch_start"] = time.time()
            elapsed = 0.0
        if launch_phase == "ram_wall_push":
            # Same "hook" idea as the general pin_phase == "wall_push"
            # above (back away, offset opposite the target push direction,
            # ram forward from that angle) instead of a straight-sideways
            # nudge - more leverage against something genuinely wedged.
            direction = search_state["pin_push_direction"]
            offset_move = "strafe_right" if direction == "strafe_left" else "strafe_left"
            subphase = search_state.get("hook_subphase", "back")
            sub_elapsed = time.time() - search_state["pin_push_start"]
            if subphase == "back":
                if sub_elapsed < PIN_HOOK_BACK_S:
                    getattr(drive, "backward")(speed=PIN_WALL_PUSH_SPEED)
                    search_state["last_move"] = "backward"
                    return f"Freeing ball {direction}: backing off ({sub_elapsed:.1f}s)"
                search_state["hook_subphase"] = "offset"
                search_state["pin_push_start"] = time.time()
                _stop(drive, search_state)
                return f"Freeing ball {direction}: repositioning"
            if subphase == "offset":
                if sub_elapsed < PIN_HOOK_OFFSET_S:
                    getattr(drive, offset_move)(speed=PIN_WALL_PUSH_SPEED)
                    search_state["last_move"] = offset_move
                    return f"Freeing ball {direction}: offsetting ({sub_elapsed:.1f}s)"
                search_state["hook_subphase"] = "ram"
                search_state["pin_push_start"] = time.time()
                _stop(drive, search_state)
                return f"Freeing ball {direction}: ramming"
            # subphase == "ram"
            if sub_elapsed < PIN_HOOK_RAM_S:
                getattr(drive, "forward")(speed=PIN_WALL_PUSH_SPEED)
                search_state["last_move"] = "forward"
                attempt = search_state["pin_push_attempt"]
                return f"Freeing ball {direction}: ram (attempt {attempt}, {sub_elapsed:.1f}s)"
            search_state.pop("hook_subphase", None)
            search_state["pin_push_attempt"] += 1
            if search_state["pin_push_attempt"] >= PIN_WALL_PUSH_ATTEMPTS:
                # Couldn't free it - give up hooking and just return to
                # post the normal way rather than repeating forever.
                search_state.pop("pin_push_start", None)
                search_state.pop("pin_push_direction", None)
                search_state.pop("pin_push_attempt", None)
                search_state["launch_phase"] = "returning"
                search_state["launch_start"] = time.time()
                _stop(drive, search_state)
                return "Couldn't free ball from wall - returning to post"
            search_state["pin_push_direction"] = (
                "strafe_right" if direction == "strafe_left" else "strafe_left"
            )
            search_state["pin_push_start"] = time.time()
            _stop(drive, search_state)
            return f"Trying other side (attempt {search_state['pin_push_attempt']})"
        return_duration = search_state.get("launch_return_duration", MAX_LAUNCH_DURATION_S)
        if elapsed < return_duration:
            if _safe_move(drive, search_state, "backward", LAUNCH_SPEED):
                return f"Returning to position ({elapsed:.1f}s)"
            return f"Braking before returning ({elapsed:.1f}s)"
        search_state.pop("launch_phase", None)
        search_state.pop("launch_start", None)
        search_state.pop("launch_return_duration", None)
        search_state.pop("launch_max_area_seen", None)
        search_state.pop("launch_last_seen", None)
        return "Launch complete - resuming defense"

    if center is None:
        # About to search - any velocity estimate from before this gap is
        # meaningless once reacquired, so drop it instead of relying on the
        # dt/kind guards in _lead_adjusted_x to catch it.
        search_state.pop("track_prev_x", None)
        search_state.pop("track_prev_t", None)
        search_state.pop("track_prev_kind", None)
        frame, masks, center = _SEARCH_MODES[SEARCH_MODE](cap, drive)
        search_state["last_move"] = "stop"  # every _search_for_ball_* variant always ends stopped
        if center is None:
            _stop(drive, search_state)
            return "No ball found (strafed L/R) - holding position"

    x, y = center
    target_kind = "ball" if target_mask is ball_mask else "blue"
    lead_x = _lead_adjusted_x(x, target_kind, search_state)
    zone = _ball_zone(lead_x, frame.shape[1])
    coord = f"(x={x}, lead_x={lead_x:.0f}, y={y})"

    if zone == "L":
        if _safe_move(drive, search_state, "strafe_left", STRAFE_SPEED):
            return f"Ball det L {coord} - moving to C (strafe left)"
        return f"Ball det L {coord} - braking before reversing to strafe left"
    if zone == "R":
        if _safe_move(drive, search_state, "strafe_right", STRAFE_SPEED):
            return f"Ball det R {coord} - moving to C (strafe right)"
        return f"Ball det R {coord} - braking before reversing to strafe right"

    # Before ramming or approaching, make sure we're actually squared up on
    # HOME_HEADING_DEG - if search drift left us rotated off-heading,
    # driving "forward" now would push the ball off at an angle instead of
    # straight out, which near our own goal risks deflecting it back in.
    # Correct heading first and hold off on any forward motion until it's
    # within tolerance.
    heading_diff = _heading_diff(drive)
    if abs(heading_diff) > STRAIGHT_HEADING_TOLERANCE_DEG:
        drive.rotate_to(HOME_HEADING_DEG)
        search_state["last_move"] = "stop"
        return f"Ball det C {coord} - correcting heading ({heading_diff:+.0f} deg) before advancing"

    # Centered - launch once it's close enough (read straight off the ROI:
    # the ball's y-position in frame), otherwise just hold.
    if y >= CLOSE_Y_THRESHOLD:
        search_state["launch_phase"] = "forward"
        search_state["launch_start"] = time.time()
        search_state["launch_max_area_seen"] = False
        search_state["launch_last_seen"] = time.time()
        search_state.pop("ram_prev_gray", None)
        search_state.pop("ram_static_streak", None)
        _safe_move(drive, search_state, "forward", LAUNCH_SPEED)
        return f"Ball det C {coord} - close, launching forward"

    # Not close enough to ram yet, but heading's locked - approach instead
    # of holding still, ramping speed up from APPROACH_MIN_SPEED toward
    # LAUNCH_SPEED as the ball nears CLOSE_Y_THRESHOLD.
    approach_ratio = max(0.0, min(1.0, y / CLOSE_Y_THRESHOLD))
    approach_speed = int(
        APPROACH_MIN_SPEED + approach_ratio * (LAUNCH_SPEED - APPROACH_MIN_SPEED)
    )
    if _safe_move(drive, search_state, "forward", approach_speed):
        return f"Ball det C {coord} - approaching (speed={approach_speed})"
    return f"Ball det C {coord} - braking before approaching"


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
