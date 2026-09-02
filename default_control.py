import sys
import termios
import tty
import select

from movement import MecanumDrive


def get_key(timeout=0.1):
    # Non-blocking single keypress read from the terminal (no Enter needed)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    drive = MecanumDrive()
    print("W/S forward/back, A/D strafe, Q/E rotate, SPACE stop, X to quit.")
    print("1/2/3/4 = spin FL/FR/RL/RR alone, for wiring calibration.")
    try:
        while True:
            key = get_key(0.1)
            if key is None:
                continue
            key = key.lower()
            if key == "w":
                print("Forward")
                drive.forward()
            elif key == "s":
                print("Backward")
                drive.backward()
            elif key == "a":
                print("Strafe left")
                drive.strafe_left()
            elif key == "d":
                print("Strafe right")
                drive.strafe_right()
            elif key == "q":
                print("Rotate left")
                drive.rotate_left()
            elif key == "e":
                print("Rotate right")
                drive.rotate_right()
            elif key == "1":
                print("Testing FL")
                drive.test_wheel("fl")
            elif key == "2":
                print("Testing FR")
                drive.test_wheel("fr")
            elif key == "3":
                print("Testing RL")
                drive.test_wheel("rl")
            elif key == "4":
                print("Testing RR")
                drive.test_wheel("rr")
            elif key == " ":
                print("Stop")
                drive.stop()
            elif key == "x":
                print("Quitting...")
                break
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("Cleaning up GPIO resources...")
        drive.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
