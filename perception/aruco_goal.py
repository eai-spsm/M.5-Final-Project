import cv2

# PLACEHOLDER tag IDs - set these to whatever ArUco IDs are actually printed
# on each goal. Distinguishing by ID (not just "a tag exists") matters:
# driving into OUR_GOAL_ID scores for the opponent, not us.
OUR_GOAL_ID = 0
OPPONENT_GOAL_ID = 1

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# OpenCV changed the aruco API between versions (class-based vs a free
# function) - support both rather than guessing which one the Pi has.
if hasattr(cv2.aruco, "ArucoDetector"):
    _detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def _detect(gray):
        corners, ids, _ = _detector.detectMarkers(gray)
        return corners, ids
else:
    _aruco_params = cv2.aruco.DetectorParameters_create()

    def _detect(gray):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=_aruco_params)
        return corners, ids


def detect_markers(frame):
    """Returns {tag_id: {"center": (x, y), "corners": 4x2 array, "area": px^2}}
    for every ArUco tag found in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect(gray)

    markers = {}
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            pts = corner[0]  # 4x2 array of the tag's corner points
            center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            area = float(cv2.contourArea(pts))
            markers[int(marker_id)] = {"center": center, "corners": pts, "area": area}
    return markers


def find_goal(frame, target_id):
    """Returns that tag's {"center", "corners", "area"} dict, or None if not
    visible this frame."""
    return detect_markers(frame).get(target_id)


def draw_markers(frame, markers):
    # Debug overlay: outline + ID label for every detected tag.
    for marker_id, info in markers.items():
        pts = info["corners"].astype(int)
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        cx, cy = map(int, info["center"])
        cv2.putText(frame, f"id={marker_id}", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return frame
