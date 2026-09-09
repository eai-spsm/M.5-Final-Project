import RPi.GPIO as GPIO
from time import sleep

from guidance import GuidedDrive
from ball_chase import open_camera, chase_loop

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21

# No button wired up right now - set back to True once it is.
WAIT_FOR_BUTTON = False

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
if WAIT_FOR_BUTTON:
    GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def main():
    drive = GuidedDrive()
    cap = None
    try:
        if WAIT_FOR_BUTTON:
            print("Ready. Press the button to start...")
            while True:
                if GPIO.input(BTN_PIN) == GPIO.LOW:
                    print("Button pressed - starting...")
                    break
                sleep(0.05)

        cap = open_camera()
        if cap is None:
            return

        print("Chasing the ball. Ctrl+C to stop.")
        chase_loop(cap, drive)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    finally:
        print("Cleaning up GPIO resources...")
        drive.stop()
        drive.cleanup()
        if cap is not None:
            cap.release()
        print("Done!")


if __name__ == "__main__":
    main()
