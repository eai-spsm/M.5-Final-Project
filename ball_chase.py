import cv2

from guidance import GuidedDrive
from perception import get_masks, ball_center, ball_angle_offset

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Don't bother turning for offsets smaller than this - avoids twitching
# back and forth over tiny, noisy angle estimates when the ball's roughly
# ahead already.
CENTERED_TOLERANCE_DEG = 8


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


def chase_step(cap, drive, on_frame=None):
    # Reads one frame and reacts to it: turn toward the ball, drive at it
    # once centered, or stop if it's not visible. Returns a short status
    # string for the caller to display.
    #
    # on_frame, if given, is called with (frame, masks) for every frame -
    # e.g. to push a live view somewhere - without a second camera read
    # (only one process/reader can hold a webcam open at a time).
    ret, frame = cap.read()
    if not ret:
        return "Camera read failed"

    masks = get_masks(frame)
    if on_frame is not None:
        on_frame(frame, masks)

    center = ball_center(masks["ball"])

    if center is None:
        drive.stop()
        return "Searching..."

    angle = ball_angle_offset(center[0], frame.shape[1])
    if abs(angle) > CENTERED_TOLERANCE_DEG:
        target = (drive.pose()[2] + angle) % 360
        drive.rotate_to(target)
        return f"Ball at {angle:+5.1f} deg - turning"
    else:
        drive.forward()
        return f"Ball centered ({angle:+5.1f} deg) - approaching"


def chase_loop(cap, drive, on_frame=None):
    # Blocks forever, reacting to the ball frame by frame. Caller handles
    # KeyboardInterrupt/cleanup.
    while True:
        status = chase_step(cap, drive, on_frame=on_frame)
        print(f"\r{status:<45}", end="", flush=True)


def main():
    cap = open_camera()
    if cap is None:
        return

    drive = GuidedDrive()
    print("Ball-chase demo. Turns toward the ball, drives forward once centered.")
    print("Ctrl+C to stop.")
    try:
        chase_loop(cap, drive)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        drive.stop()
        drive.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
