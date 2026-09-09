import cv2
import numpy as np

# PLACEHOLDER HSV ranges - tune these on-site under the real match lighting.
# OpenCV hue is 0-179 (not 0-359). To find good ranges: run cam_control.py,
# open the segmentation view in the browser, and adjust these numbers while
# watching it until each target is cleanly picked out with minimal noise.
BALL_LOW = np.array([35, 80, 60])     # bright green ball
BALL_HIGH = np.array([85, 255, 255])

WALL_LOW = np.array([0, 0, 0])        # black wall/border - pure black through worn/charcoal-black
WALL_HIGH = np.array([179, 255, 90])

FLOOR_LOW = np.array([0, 0, 110])     # gray->white tile floor (low saturation)
FLOOR_HIGH = np.array([179, 60, 255])

# PLACEHOLDER - typical USB webcams are ~60-70 deg horizontal FOV, but this
# varies a lot by camera. To calibrate: point the camera at two marks a
# known angle apart (e.g. 30 deg using a protractor/marked floor), measure
# their pixel x-distance apart in the frame, and solve
# CAMERA_HFOV_DEG = known_angle * frame_width / pixel_distance_between_marks.
CAMERA_HFOV_DEG = 60.0


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


def ball_angle_offset(center_x, frame_width, fov_deg=CAMERA_HFOV_DEG):
    # How many degrees off-center the ball is, assuming a simple pinhole
    # model (no lens-distortion correction - fine near the center of the
    # frame, gets less accurate toward the edges on a wide/fisheye lens).
    # Positive = ball is to the right of center, negative = left.
    offset_fraction = (center_x - frame_width / 2) / (frame_width / 2)
    return offset_fraction * (fov_deg / 2)


# A goal opening isn't a different color from the wall - it's the same
# black, just arranged differently vertically. A normal wall section has
# TWO separate black bands per column: a lower rail near the ground, an
# open white gap above it, then an upper rail - 2 separate black runs
# going up the column. A goal only has the top crossbar (the lower rail
# is absent, open floor/background all the way down beneath it) - 1
# black run. This is far more specific than just "wall looks shorter
# here", which also fires on plain perspective/lighting variation.
GOAL_GAP_MIN_WIDTH_PX = 30  # ignore narrow single-column noise/joints


def _vertical_run_count(column):
    # How many separate contiguous "on" (wall) runs going down this one
    # column of the mask.
    runs = 0
    in_run = False
    for v in column:
        on = v > 0
        if on and not in_run:
            runs += 1
        in_run = on
    return runs


def find_goal_gap(wall_mask, min_width_px=GOAL_GAP_MIN_WIDTH_PX):
    """Looks for a goal-width stretch of columns that have only ONE
    vertical black band (crossbar-only) instead of the normal wall's two
    (lower rail + upper rail with a gap between). Returns
    {"center_x", "width_px"} for the widest qualifying stretch, or None.
    """
    height, width = wall_mask.shape
    is_single_bar = np.zeros(width, dtype=bool)
    for x in range(width):
        column = wall_mask[:, x]
        if not column.any():
            continue  # no wall visible in this column at all - not a crossbar-only signal
        is_single_bar[x] = _vertical_run_count(column) == 1

    # Find the widest contiguous run of single-bar columns.
    best_start, best_len = None, 0
    run_start = None
    for x, single in enumerate(is_single_bar):
        if single:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None:
                run_len = x - run_start
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
                run_start = None
    if run_start is not None:  # run reached the right edge of the frame
        run_len = width - run_start
        if run_len > best_len:
            best_start, best_len = run_start, run_len

    if best_start is None or best_len < min_width_px:
        return None

    return {"center_x": best_start + best_len / 2, "width_px": best_len}


def build_debug_view(frame, masks=None):
    # 2x2 grid: ball detection (annotated) | ball cut-out
    #           wall cut-out               | floor cut-out
    # Pass masks in if the caller already has them (from get_masks) to
    # avoid recomputing.
    if masks is None:
        masks = get_masks(frame)

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
