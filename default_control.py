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


def print_status(action, drive):
    x, y, heading = drive.pose()
    print(f"\r{action:<14} x={x:7.1f}cm y={y:7.1f}cm heading={heading:5.1f}°   ", end="", flush=True)


def main():
    drive = GuidedDrive()
    print("W/S forward/back, A/D strafe, Q/E rotate, SPACE stop, X to quit.")
    print("1/2/3/4 = spin FL/FR/RL/RR alone, for wiring calibration.")
    print("B = about-face (rotate 180 from current heading).")
    print("R = reset tracked position to (0, 0), heading 0.")
    print("+/- = adjust speed.")
    print("H = HALT (locks out other keys until H is pressed again).")
    action = "Ready"
    halted = False
    try:
        while True:
            key = get_key(0.1)
            if key is not None:
                key = key.lower()
                if key == "h":
                    halted = not halted
                    drive.stop()
                    action = "HALTED (press H to resume)" if halted else "Resumed"
                elif key == "x":
                    print("\nQuitting...")
                    break
                elif halted:
                    pass  # ignore every other key while halted
                elif key == "w":
                    drive.forward()
                    action = "Forward"
                elif key == "s":
                    drive.backward()
                    action = "Backward"
                elif key == "a":
                    drive.strafe_left()
                    action = "Strafe left"
                elif key == "d":
                    drive.strafe_right()
                    action = "Strafe right"
                elif key == "q":
                    drive.rotate_left()
                    action = "Rotate left"
                elif key == "e":
                    drive.rotate_right()
                    action = "Rotate right"
                elif key == "1":
                    drive.test_wheel("fl")
                    action = "Testing FL"
                elif key == "2":
                    drive.test_wheel("fr")
                    action = "Testing FR"
                elif key == "3":
                    drive.test_wheel("rl")
                    action = "Testing RL"
                elif key == "4":
                    drive.test_wheel("rr")
                    action = "Testing RR"
                elif key == "b":
                    target = (drive.pose()[2] + 180) % 360
                    drive.rotate_to(target)
                    action = "About-face"
                elif key == "r":
                    drive.reset_position()
                    action = "Position reset"
                elif key in ("+", "="):
                    speed, strafe_speed = drive.adjust_speed(5)
                    action = f"Speed {speed:.0f}/{strafe_speed:.0f}"
                elif key == "-":
                    speed, strafe_speed = drive.adjust_speed(-5)
                    action = f"Speed {speed:.0f}/{strafe_speed:.0f}"
                elif key == " ":
                    drive.stop()
                    action = "Stop"
            print_status(action, drive)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("\nCleaning up GPIO resources...")
        drive.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
