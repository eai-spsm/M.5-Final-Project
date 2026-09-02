import RPi.GPIO as GPIO
from time import sleep

from movement import MecanumDrive

# Start button (pulled up, wired to GND when pressed)
BTN_PIN = 21

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BTN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def main():
    drive = MecanumDrive()
    try:
        print("Ready. Press the button to start...")
        while True:
            if GPIO.input(BTN_PIN) == GPIO.LOW:
                print("Button pressed - starting...")
                break
            sleep(0.05)

        # TODO: put your startup logic here (e.g. perception + drive loop)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

    finally:
        print("Cleaning up GPIO resources...")
        drive.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
