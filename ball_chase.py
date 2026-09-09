import cv2

from guidance import GuidedDrive
from perception import get_masks, ball_center, ball_angle_offset

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Don't bother turning for offsets smaller than this - avoids twitching
# back and forth over tiny, noisy angle estimates when the ball's roughly
# ahead already.
CENTERED_TOLERANCE_DEG = 8


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    drive = GuidedDrive()
    print("Ball-chase demo. Turns toward the ball, drives forward once centered.")
    print("Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            masks = get_masks(frame)
            center = ball_center(masks["ball"])

            if center is None:
                drive.stop()
                print("\rSearching...                                        ", end="", flush=True)
                continue

            angle = ball_angle_offset(center[0], frame.shape[1])
            if abs(angle) > CENTERED_TOLERANCE_DEG:
                target = (drive.pose()[2] + angle) % 360
                drive.rotate_to(target)
                print(f"\rBall at {angle:+5.1f} deg - turning                  ", end="", flush=True)
            else:
                drive.forward()
                print(f"\rBall centered ({angle:+5.1f} deg) - approaching       ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        drive.stop()
        drive.cleanup()
        cap.release()


if __name__ == "__main__":
    main()
