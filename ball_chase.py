import time

import cv2

from guidance import GuidedDrive
from perception import get_masks, ball_center, ball_angle_offset, find_goal_gap

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Don't bother turning for offsets smaller than this - avoids twitching
# back and forth over tiny, noisy angle estimates when the ball's roughly
# ahead already.
CENTERED_TOLERANCE_DEG = 8

# Duty cycle % used while spinning to search for the ball - slower than a
# normal turn so a frame doesn't blur past the ball and miss it.
SEARCH_SPEED = 40

# If the wall mask covers this much of the frame, we're facing straight
# into it (or nearly touching it) - back up instead of trying to chase
# whatever the ball logic thinks it sees.
WALL_COVERAGE_THRESHOLD = 0.80

# If something's this close according to the ultrasonic AND it's not the
# wall (that's handled separately above, by the vision check), evade it -
# doesn't matter what it actually is (opponent robot, dropped object,
# anything), the ultrasonic can't tell and doesn't need to.
EVADE_DISTANCE_CM = 15
EVADE_SPEED = 45

# Don't linger in front of a goal opening (either one) for longer than
# this - matches the rulebook's forbidden-zone-near-goal idea. Proximity
# is gated on the ultrasonic (GOAL_AREA_DISTANCE_CM) so a goal merely
# visible far across the field doesn't start the clock.
GOAL_AREA_DISTANCE_CM = 30
GOAL_DWELL_LIMIT_S = 5.0

# PLACEHOLDER - confirm against the real field/starting setup. World-frame
# heading (Navigator's convention: 0 = wherever the robot was facing at
# the start) that the OPPONENT's goal is roughly in the direction of. Used
# to guess which goal a detected gap belongs to when there's no ArUco tag
# (or it's not legible) to say for certain - see guess_goal_ownership().
OPPONENT_GOAL_HEADING_DEG = 0.0

# After spinning this many degrees without finding the ball, assume it's
# not visible from here (behind something, out of view) and nudge forward
# before continuing the search, instead of spinning in the same spot
# forever. Slightly under 360 to account for tracking imprecision.
SEARCH_FULL_SWEEP_DEG = 350

# If the camera read fails this many times in a row, assume it's actually
# disconnected (not just a one-off dropped frame) and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0


def guess_goal_ownership(current_heading_deg, gap_bearing_deg, opponent_goal_heading_deg=OPPONENT_GOAL_HEADING_DEG):
    # A detected wall gap only says "there's an opening here", not which
    # goal it is - reconcile using the tracked heading. Converts the gap's
    # frame-relative bearing into an absolute world-frame direction, then
    # checks which goal's assumed heading it's closer to (+-90 deg = same
    # half of the compass). Prefer an ArUco tag ID when one's visible
    # (perception.aruco_goal) - that's ground truth; this is a fallback
    # guess for when a tag isn't visible/legible.
    world_bearing = (current_heading_deg + gap_bearing_deg) % 360
    diff = (world_bearing - opponent_goal_heading_deg + 180) % 360 - 180  # -180..180
    return "opponent" if abs(diff) <= 90 else "own"


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
    # Reads one frame and reacts to it: turn toward the ball, drive at it
    # once centered, back up if we're right up against the wall, or search
    # if it's not visible. Returns a short status string for the caller to
    # display.
    #
    # on_frame, if given, is called with (frame, masks) for every frame -
    # e.g. to push a live view somewhere - without a second camera read
    # (only one process/reader can hold a webcam open at a time).
    #
    # search_state, if given, is a dict chase_loop persists across calls so
    # the search sweep can track how far it's turned. Without it (e.g.
    # called standalone/in a test) each call starts a fresh sweep.
    if search_state is None:
        search_state = {}

    ret, frame = cap.read()
    if not ret:
        # Don't leave the last command running blind if the camera drops.
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    if on_frame is not None:
        on_frame(frame, masks)

    wall_fraction = float((masks["wall"] > 0).mean())
    if wall_fraction >= WALL_COVERAGE_THRESHOLD:
        drive.backward()
        search_state["searching"] = False
        return f"Wall fills {wall_fraction * 100:.0f}% of view - backing up"

    distance = drive.get_distance()

    # Don't overstay near a goal opening (rule compliance, not physical
    # safety like the wall/evade checks) - track how long we've been
    # close to one and force a retreat past GOAL_DWELL_LIMIT_S. Doesn't
    # need to know which goal (see guess_goal_ownership() for that,
    # used elsewhere for not shooting into our own goal).
    gap = find_goal_gap(masks["wall"])
    near_goal = gap is not None and distance is not None and distance < GOAL_AREA_DISTANCE_CM
    if near_goal:
        if "goal_area_since" not in search_state:
            search_state["goal_area_since"] = time.time()
        dwell = time.time() - search_state["goal_area_since"]
        if dwell >= GOAL_DWELL_LIMIT_S:
            drive.backward()
            search_state["searching"] = False
            return f"In goal area {dwell:.1f}s - leaving"
    else:
        search_state.pop("goal_area_since", None)

    center = ball_center(masks["ball"])
    angle = ball_angle_offset(center[0], frame.shape[1]) if center is not None else None
    if angle is not None:
        # Remembered even when we're about to evade instead of chase, so an
        # evade triggered the moment the ball gets blocked can still evade
        # toward the side it was last seen on.
        search_state["last_ball_angle"] = angle

    if distance is not None and distance < EVADE_DISTANCE_CM:
        # Not the wall (that's already handled above) - something else is
        # right in front of us, most likely between us and the ball. Go
        # around whichever side the ball actually is (or was last seen
        # on, if it's hidden behind whatever's blocking us right now)
        # instead of blindly picking a direction.
        last_angle = search_state.get("last_ball_angle", 0)
        if last_angle < 0:
            drive.strafe_left(speed=EVADE_SPEED)
            direction = "left"
        else:
            drive.strafe_right(speed=EVADE_SPEED)
            direction = "right"
        search_state["searching"] = False
        return f"Obstruction at {distance:.0f}cm - evading {direction} toward ball"

    if center is None:
        heading = drive.pose()[2]
        if not search_state.get("searching"):
            search_state["searching"] = True
            search_state["swept_deg"] = 0.0
        else:
            # Accumulate the actual step-to-step rotation rather than
            # comparing against a fixed start heading - a single mod-360
            # comparison can skip right past the threshold if a step
            # happens to cross the 0/360 wrap between polls.
            delta = (heading - search_state["last_heading"] + 180) % 360 - 180
            search_state["swept_deg"] += abs(delta)
        search_state["last_heading"] = heading

        drive.rotate_right(speed=SEARCH_SPEED)
        swept = search_state["swept_deg"]

        if swept >= SEARCH_FULL_SWEEP_DEG:
            drive.forward(speed=SEARCH_SPEED)
            search_state["searching"] = False  # restarts the sweep next call
            return "Full 360 sweep, no ball found - repositioning..."

        return f"Searching (spinning, {swept:.0f}/360 deg)..."

    search_state["searching"] = False

    if abs(angle) > CENTERED_TOLERANCE_DEG:
        target = (drive.pose()[2] + angle) % 360
        drive.rotate_to(target)
        return f"Ball at {angle:+5.1f} deg - turning"
    else:
        drive.forward()
        return f"Ball centered ({angle:+5.1f} deg) - approaching"


def _safe_print(*args, **kwargs):
    # If stdout is gone (e.g. an SSH session dropped), printing raises
    # BrokenPipeError/OSError - that's fine to ignore, the driving logic
    # doesn't need anyone watching to keep working.
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def chase_loop(cap, drive, on_frame=None, on_reconnect=None):
    # Blocks forever, reacting to the ball frame by frame. Caller handles
    # KeyboardInterrupt/cleanup. Survives the camera dropping out - keeps
    # the motors stopped and retries opening it rather than crashing or
    # driving blind on stale commands.
    #
    # on_reconnect, if given, is called with the new capture object each
    # time the camera is reopened - the caller's own `cap` variable (used
    # for cleanup) would otherwise go stale, since reassigning the local
    # `cap` here doesn't change what the caller is holding.
    consecutive_failures = 0
    search_state = {}
    while True:
        status = chase_step(cap, drive, on_frame=on_frame, search_state=search_state)

        if status == "Camera read failed":
            consecutive_failures += 1
            if consecutive_failures >= CAMERA_RECONNECT_AFTER:
                _safe_print(f"\r{'Camera lost - reconnecting...':<45}", end="", flush=True)
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

        _safe_print(f"\r{status:<45}", end="", flush=True)


def main():
    cap = open_camera()
    if cap is None:
        return

    def _track_cap(new_cap):
        nonlocal cap
        cap = new_cap

    drive = GuidedDrive()
    print("Ball-chase demo. Turns toward the ball, drives forward once centered.")
    print("Ctrl+C to stop.")
    try:
        chase_loop(cap, drive, on_reconnect=_track_cap)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        drive.stop()
        drive.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
