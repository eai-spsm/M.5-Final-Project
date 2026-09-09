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
