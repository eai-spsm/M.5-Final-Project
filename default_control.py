import RPi.GPIO as GPIO
import sys
import termios
import tty
import select
import time

# 1. Tell the Pi we are using BCM (GPIO labels), not physical board numbers
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Orientation: battery = North (front), Raspberry Pi = South (back)

# Board 1 - Left side
ENA_L = 12  # enables IN1/IN2
ENB_L = 18  # enables IN3/IN4
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22

# Board 2 - Right side
ENA_R = 13  # enables IN5/IN6
ENB_R = 23  # enables IN7/IN8
IN5 = 5
IN6 = 6
IN7 = 19
IN8 = 26

# Ultrasonic (HC-SR04)
TRIG = 10
ECHO = 9

GPIO.setup(ENA_L, GPIO.OUT)
GPIO.setup(ENB_L, GPIO.OUT)
GPIO.setup(ENA_R, GPIO.OUT)
GPIO.setup(ENB_R, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)
GPIO.setup(IN7, GPIO.OUT)
GPIO.setup(IN8, GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, GPIO.LOW)

pwm_left_a = GPIO.PWM(ENA_L, 1000)   # IN1/IN2
pwm_left_b = GPIO.PWM(ENB_L, 1000)   # IN3/IN4
pwm_right_a = GPIO.PWM(ENA_R, 1000)  # IN5/IN6
pwm_right_b = GPIO.PWM(ENB_R, 1000)  # IN7/IN8
pwm_left_a.start(0)
pwm_left_b.start(0)
pwm_right_a.start(0)
pwm_right_b.start(0)

SPEED = 60  # duty cycle %
# Strafing runs each board's two channels in opposite directions at once,
# which on cheap L298N-style boards can couple electrical noise between
# channels and glitch the H-bridge. Lower speed = lower current/back-EMF.
STRAFE_SPEED = 55  # duty cycle %
OBSTACLE_CM = 15  # minimum clearance before forward drive is blocked

# If W drives diagonally instead of straight, one wheel is physically wired
# backward. Use the "1"/"2"/"3"/"4" calibration keys to find which one, then
# flip its flag here.
WHEEL_INVERT = {"fl": False, "fr": False, "rl": False, "rr": True}

# If strafing (A/D) drifts at an angle or spins instead of going straight
# sideways, the wheels aren't spinning at exactly matched speeds even at the
# same duty cycle - lower the trim on whichever wheel is "winning" until it
# cancels out.
WHEEL_TRIM = {"fl": 0.9, "fr": 0.9, "rl": 1.0, "rr": 1.0}


def get_distance():
    # Returns distance in cm, or None if the sensor timed out
    try:
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == GPIO.LOW:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        timeout = time.time() + 0.04
        while GPIO.input(ECHO) == GPIO.HIGH:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        return (pulse_end - pulse_start) * 34300 / 2
    except Exception as e:
        print(f"Ultrasonic Read Error: {e}")
        return None


# Each wheel is driven independently so we can strafe/rotate, not just go left/right.
# FL = IN1/IN2 (ENA_L), RL = IN3/IN4 (ENB_L), FR = IN7/IN8 (ENB_R), RR = IN5/IN6 (ENA_R)
# Left and right boards each have their own, DIFFERENT front/rear channel
# order (confirmed with the 1/2/3/4 calibration keys - do not assume they
# match; re-check with those keys after any wiring change).

def _wheel(pin_a, pin_b, forward):
    GPIO.output(pin_a, GPIO.LOW if forward else GPIO.HIGH)
    GPIO.output(pin_b, GPIO.HIGH if forward else GPIO.LOW)


def _wheel_stop(pin_a, pin_b):
    GPIO.output(pin_a, GPIO.LOW)
    GPIO.output(pin_b, GPIO.LOW)


def set_wheels(fl, fr, rl, rr, speed=SPEED):
    # Each arg is 1 (forward), -1 (backward), or 0 (stop)
    wheels = {"fl": fl, "fr": fr, "rl": rl, "rr": rr}
    pins = {"fl": (IN1, IN2), "fr": (IN7, IN8), "rl": (IN3, IN4), "rr": (IN5, IN6)}
    pwms = {"fl": pwm_left_a, "fr": pwm_right_b, "rl": pwm_left_b, "rr": pwm_right_a}

    for name, direction in wheels.items():
        pin_a, pin_b = pins[name]
        if direction == 0:
            _wheel_stop(pin_a, pin_b)
            pwms[name].ChangeDutyCycle(0)
        else:
            if WHEEL_INVERT[name]:
                direction = -direction
            _wheel(pin_a, pin_b, forward=direction > 0)
            duty = min(100, max(0, speed * WHEEL_TRIM[name]))
            pwms[name].ChangeDutyCycle(duty)


def test_wheel(name):
    # Spins one wheel forward in isolation, for calibrating WHEEL_INVERT
    directions = {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
    directions[name] = 1
    set_wheels(**directions)


def stop_all():
    set_wheels(0, 0, 0, 0)


def drive_forward():
    distance = get_distance()
    if distance is not None and distance < OBSTACLE_CM:
        print(f"Obstacle at {distance:.1f}cm - forward blocked")
        stop_all()
        return
    set_wheels(1, 1, 1, 1)


def drive_backward():
    set_wheels(-1, -1, -1, -1)


def strafe_left():
    set_wheels(-1, 1, 1, -1, speed=STRAFE_SPEED)


def strafe_right():
    set_wheels(1, -1, -1, 1, speed=STRAFE_SPEED)


def rotate_left():
    set_wheels(-1, 1, -1, 1)


def rotate_right():
    set_wheels(1, -1, 1, -1)


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
                drive_forward()
            elif key == "s":
                print("Backward")
                drive_backward()
            elif key == "a":
                print("Strafe left")
                strafe_left()
            elif key == "d":
                print("Strafe right")
                strafe_right()
            elif key == "q":
                print("Rotate left")
                rotate_left()
            elif key == "e":
                print("Rotate right")
                rotate_right()
            elif key == "1":
                print("Testing FL")
                test_wheel("fl")
            elif key == "2":
                print("Testing FR")
                test_wheel("fr")
            elif key == "3":
                print("Testing RL")
                test_wheel("rl")
            elif key == "4":
                print("Testing RR")
                test_wheel("rr")
            elif key == " ":
                print("Stop")
                stop_all()
            elif key == "x":
                print("Quitting...")
                break
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("Cleaning up GPIO resources...")
        stop_all()
        pwm_left_a.stop()
        pwm_left_b.stop()
        pwm_right_a.stop()
        pwm_right_b.stop()
        GPIO.cleanup()
        print("Done!")


if __name__ == "__main__":
    main()
