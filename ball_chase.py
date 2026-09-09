import time

import cv2

from guidance import GuidedDrive
from perception import get_masks, ball_center, ball_angle_offset

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Don't bother turning for offsets smaller than this - avoids twitching
# back and forth over tiny, noisy angle estimates when the ball's roughly
# ahead already.
CENTERED_TOLERANCE_DEG = 8

# Duty cycle % used while spinning to search for the ball - slower than a
# normal turn so a frame doesn't blur past the ball and miss it.
SEARCH_SPEED = 40

# If the camera read fails this many times in a row, assume it's actually
# disconnected (not just a one-off dropped frame) and try to reopen it.
CAMERA_RECONNECT_AFTER = 20
CAMERA_RECONNECT_RETRY_DELAY = 1.0


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
        # Don't leave the last command running blind if the camera drops.
        drive.stop()
        return "Camera read failed"

    masks = get_masks(frame)
    if on_frame is not None:
        on_frame(frame, masks)

    center = ball_center(masks["ball"])

    if center is None:
        # Not found - spin in place (continues however far around it takes,
        # "360" isn't tracked/enforced, it just keeps going until a frame
        # finds the ball) rather than sitting stopped and blind.
        drive.rotate_right(speed=SEARCH_SPEED)
        return "Searching (spinning)..."

    angle = ball_angle_offset(center[0], frame.shape[1])
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
    while True:
        status = chase_step(cap, drive, on_frame=on_frame)

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
