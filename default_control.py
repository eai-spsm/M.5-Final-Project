import sys
import termios
import tty
import select

from guidance import GuidedDrive


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
    drive = GuidedDrive()
    print("W/S forward/back, A/D strafe, Q/E rotate, SPACE stop, X to quit.")
    print("1/2/3/4 = spin FL/FR/RL/RR alone, for wiring calibration.")
    print("P = print current position/heading. Starts at (0, 0), heading 0.")
    try:
        while True:
            key = get_key(0.1)
            if key is None:
                continue
            key = key.lower()
            if key == "w":
                drive.forward()
                print("Forward", drive.pose())
            elif key == "s":
                drive.backward()
                print("Backward", drive.pose())
            elif key == "a":
                drive.strafe_left()
                print("Strafe left", drive.pose())
            elif key == "d":
                drive.strafe_right()
                print("Strafe right", drive.pose())
            elif key == "q":
                drive.rotate_left()
                print("Rotate left", drive.pose())
            elif key == "e":
                drive.rotate_right()
                print("Rotate right", drive.pose())
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
            elif key == "p":
                print("Position:", drive.pose())
            elif key == " ":
                drive.stop()
                print("Stop", drive.pose())
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
