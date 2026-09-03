import sys
import termios
import time
import tty
import select

from guidance import GuidedDrive

DRIVE_KEYS = {"w", "s", "a", "d", "q", "e", "b"}


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


class Timer:
    """Starts on the first drive command, pauses/resumes with Halt, and
    stops for good on quit (X or Ctrl+C)."""

    def __init__(self):
        self._segment_start = None  # time.time() when the current run began, None if not running
        self._elapsed = 0.0         # accumulated seconds from finished segments

    def start_if_needed(self):
        if self._segment_start is None and self._elapsed == 0.0:
            self._segment_start = time.time()

    def pause(self):
        if self._segment_start is not None:
            self._elapsed += time.time() - self._segment_start
            self._segment_start = None

    def resume(self):
        if self._segment_start is None and self._elapsed > 0.0:
            self._segment_start = time.time()

    def stop(self):
        self.pause()

    def elapsed(self):
        if self._segment_start is not None:
            return self._elapsed + (time.time() - self._segment_start)
        return self._elapsed


def print_status(action, drive, timer):
    x, y, heading = drive.pose()
    print(f"\r{action:<28} t={timer.elapsed():6.1f}s x={x:7.1f}cm y={y:7.1f}cm heading={heading:5.1f}°   ", end="", flush=True)


def main():
    drive = GuidedDrive()
    timer = Timer()
    print("W/S forward/back, A/D strafe, Q/E rotate, SPACE stop, X to quit.")
    print("1/2/3/4 = spin FL/FR/RL/RR alone, for wiring calibration.")
    print("B = about-face (rotate 180 from current heading).")
    print("R = reset tracked position to (0, 0), heading 0.")
    print("+/- = adjust speed.")
    print("H = HALT (locks out other keys until H is pressed again).")
    print("Timer starts on the first drive command, pauses on Halt, stops on quit.")
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
                    if halted:
                        timer.pause()
                        action = "HALTED (press H to resume)"
                    else:
                        timer.resume()
                        action = "Resumed"
                elif key == "x":
                    print("\nQuitting...")
                    break
                elif halted:
                    pass  # ignore every other key while halted
                elif key == "w":
                    drive.forward()
                    timer.start_if_needed()
                    action = "Forward"
                elif key == "s":
                    drive.backward()
                    timer.start_if_needed()
                    action = "Backward"
                elif key == "a":
                    drive.strafe_left()
                    timer.start_if_needed()
                    action = "Strafe left"
                elif key == "d":
                    drive.strafe_right()
                    timer.start_if_needed()
                    action = "Strafe right"
                elif key == "q":
                    drive.rotate_left()
                    timer.start_if_needed()
                    action = "Rotate left"
                elif key == "e":
                    drive.rotate_right()
                    timer.start_if_needed()
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
                    timer.start_if_needed()
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
            print_status(action, drive, timer)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        timer.stop()
        print(f"\nTotal drive time: {timer.elapsed():.1f}s")
        print("Cleaning up GPIO resources...")
        drive.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
